import argparse
import random
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import pydicom
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
from augmentations import RSNAAugmentation
from datasets import RSNADataset
from utils.general_utils import omegaconf_to_yaml


def inference(config):

    seed = config['seed']
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    n_tta = config['n_tta']

    output_dir = Path(config['save_root']) / \
        config['version'] / config['model']

    if output_dir.exists():
        print('This version already exists.\n'
              f'version:{output_dir}')
        ans = None
        while ans not in ['y', 'Y']:
            ans = input('Do you want to continue inference? (y/n): ')
            if ans in ['n', 'N']:
                quit()
    output_dir.mkdir(exist_ok=True, parents=True)

    transform = RSNAAugmentation

    dataset_args = {
        'transform': transform(mode='test', **config["transform"]),
        **config["dataset"]
    }
    dataset = RSNADataset(**dataset_args)
    loader = DataLoader(dataset=dataset,
                        batch_size=config["batch_size"],
                        shuffle=False,
                        num_workers=config["n_workers"],
                        pin_memory=False)

    device = config['gpu'][0]

    checkpoint_list = config['checkpoint']

    net_list = []
    for ckpt_path_dict in checkpoint_list:
        model_args = {}
        ckpt_path_cnn = Path(ckpt_path_dict['cnn'])
        cfg_path = ckpt_path_cnn.parents[1] / 'train_config.yaml'
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
        model_args['cnn_model'] = cfg["model"]["name"]
        model_args['cnn_pretrained_path'] = ckpt_path_cnn
        model_args['cnn_param'] = cfg["model"]["args"]
        model_args['cnn_param']['pretrained'] = False

        ckpt_path_cnn = Path(ckpt_path_dict['rnn'])
        cfg_path = ckpt_path_cnn.parents[1] / 'train_config.yaml'
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
        model_args['rnn_model'] = cfg["model"]["name"]
        model_args['rnn_pretrained_path'] = ckpt_path_cnn
        model_args['rnn_param'] = cfg["model"]["args"]

        net = getattr(models, 'CNN_RNN')(**model_args)
        net.to(device)
        net.eval()

        net_list.append(net)
    
    chunk_size_list = list(config["chunk_size_list"])
    if len(chunk_size_list) == 1:
        chunk_size_list = chunk_size_list * len(net_list)
    
    assert len(chunk_size_list) == len(net_list)
    
    tta = None

    exam_names = []
    image_names = []
    with torch.no_grad():
        results_image_level = []
        results_exam_level = []
        for x, exam_name, image_name in tqdm(loader):
            # Note: shape of exam_name & image_name.
            # exam_name -> ['exam_name']
            # image_name -> [('image_name_1', ), ..., ('image_name_n', )]
            exam_names.append(exam_name[0])
            image_name = list(map(lambda x: x[0], image_name))
            image_names.extend(image_name)
            n_sequence = x.size()[1]
            result_tta_image, result_tta_exam = [], []
            for tta_cnt in range(n_tta):
                image = x.clone()
                result_net_exam, result_net_image = [], []
                for net_cnt, net in enumerate(net_list):
                    embeddings = []
                    for i in range(0, n_sequence, chunk_size_list[net_cnt]):
                        embedding = net.cnn(image[:, i:i + chunk_size_list[net_cnt], :, :, :].to(device))
                        embeddings.append(embedding)
                    embeddings = torch.cat(embeddings, dim=1)
                    image_level, exam_level = net.rnn(embeddings)

                    image_level = torch.sigmoid(
                        image_level).cpu().detach().numpy().reshape(-1) # (sequence, )
                    exam_level = torch.sigmoid(
                        exam_level).cpu().detach().numpy().reshape(-1) #(9, )

                    exam_level = label_consistency(image_level, exam_level)
                    
                    result_net_image.append(image_level)
                    result_net_exam.append(exam_level)
                result_net_image = np.array(result_net_image) #(len(net_list), sequence)
                result_tta_image.append(result_net_image)
                result_net_exam = np.array(result_net_exam) #(len(net_list), 9)
                result_tta_exam.append(result_net_exam)
            result_tta_image = np.array(result_tta_image) #(n_tta, len(net_list), sequence)
            results_image_level.append(result_tta_image)
            result_tta_exam = np.array(result_tta_exam) #(n_tta, len(net_list), 9)
            results_exam_level.append(result_tta_exam)
    
    results_exam_level = np.array(results_exam_level) #(n_exam, n_tta, len(net_list), 9)

    # Note: shape of results_image_level. 
    # len(results_image_level) = n_exam
    # results_image_level[i].shape = (n_tta, len(net_list), #image in exam i)
    
    results_exam_level = results_exam_level.mean(axis=1).mean(axis=1)
    results_image_level = list(map(
        lambda x: x.mean(axis=0).mean(axis=0), results_image_level))
    results_exam_level = np.stack([label_consistency(image_level, exam_level)
        for image_level, exam_level in zip(results_image_level, results_exam_level)])
    results_image_level = np.concatenate(results_image_level)

    results_exam_level = results_exam_level.reshape(-1)

    exam_names = get_exam_names(exam_names)
    assert len(image_names) == len(results_image_level)
    assert len(exam_names) == len(results_exam_level)
    names = image_names + exam_names

    results = np.concatenate([results_image_level, results_exam_level])

    submission = pd.DataFrame([names, results], index=['id', 'label']).T
    
    if check_consistency(submission, dataset.df):
        print("Great! Fanstastic! You are genious!!!" )
        submission.to_csv(output_dir / 'submission.csv', index=False)
    else:
        print("ERROR! submission file doesn't satisfy concistency!!")

    with open(output_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(omegaconf_to_yaml(cfg), f)


def label_consistency(image_level, exam_level):
    p_negative_exam_for_pe = exam_level[0]
    p_indeterminate = exam_level[1]
    p_chronic_pe = exam_level[2]
    p_acute_and_chronic_pe = exam_level[3]
    p_central_pe = exam_level[4]
    p_leftsided_pe = exam_level[5]
    p_rightsided_pe = exam_level[6]
    p_rv_lv_ratio_gte_1 = exam_level[7]
    p_rv_lv_ratio_lt_1 = exam_level[8]

    pe_exist = np.any(image_level > 0.5)
    if pe_exist:
        p_negative_exam_for_pe = np.clip(p_negative_exam_for_pe, None, 0.499)
        p_indeterminate = np.clip(p_indeterminate, None, 0.499)

        if p_chronic_pe > 0.5 and p_acute_and_chronic_pe > 0.5:
            tmp_list = [p_chronic_pe, p_acute_and_chronic_pe]
            tmp_list[np.argmin(tmp_list)] = 0.499
            p_chronic_pe, p_acute_and_chronic_pe = tmp_list
        
        if p_central_pe <= 0.5 and p_leftsided_pe <= 0.5 and p_rightsided_pe <= 0.5:
            tmp_list = [p_central_pe, p_leftsided_pe, p_rightsided_pe]
            tmp_list[np.argmax(tmp_list)] = 0.501
            p_central_pe, p_leftsided_pe, p_rightsided_pe = tmp_list
        
        if p_rv_lv_ratio_gte_1 <= 0.5 and p_rv_lv_ratio_lt_1 <= 0.5:
            tmp_list = [p_rv_lv_ratio_gte_1, p_rv_lv_ratio_lt_1]
            tmp_list[np.argmax(tmp_list)] = 0.501
            p_rv_lv_ratio_gte_1, p_rv_lv_ratio_lt_1 = tmp_list
        if p_rv_lv_ratio_gte_1 > 0.5 and p_rv_lv_ratio_lt_1 > 0.5:
            tmp_list = [p_rv_lv_ratio_gte_1, p_rv_lv_ratio_lt_1]
            tmp_list[np.argmin(tmp_list)] = 0.499
            p_rv_lv_ratio_gte_1, p_rv_lv_ratio_lt_1 = tmp_list
        
    else:
        if p_negative_exam_for_pe <= 0.5 and p_indeterminate <= 0.5:
            tmp_list = [p_negative_exam_for_pe, p_indeterminate]
            tmp_list[np.argmax(tmp_list)] = 0.501
            p_negative_exam_for_pe, p_indeterminate = tmp_list
        if p_negative_exam_for_pe > 0.5 and p_indeterminate > 0.5:
            tmp_list = [p_negative_exam_for_pe, p_indeterminate]
            tmp_list[np.argmin(tmp_list)] = 0.499
            p_negative_exam_for_pe, p_indeterminate = tmp_list
        
        p_chronic_pe = np.clip(p_chronic_pe, None, 0.499)
        p_acute_and_chronic_pe = np.clip(p_acute_and_chronic_pe, None, 0.499)

        p_central_pe = np.clip(p_central_pe, None, 0.499)
        p_leftsided_pe = np.clip(p_leftsided_pe, None, 0.499)
        p_rightsided_pe = np.clip(p_rightsided_pe, None, 0.499)

        p_rv_lv_ratio_gte_1 = np.clip(p_rv_lv_ratio_gte_1, None, 0.499)
        p_rv_lv_ratio_lt_1 = np.clip(p_rv_lv_ratio_lt_1, None, 0.499)

    exam_level = np.array([
        p_negative_exam_for_pe,
        p_indeterminate,
        p_chronic_pe,
        p_acute_and_chronic_pe,
        p_central_pe,
        p_leftsided_pe,
        p_rightsided_pe,
        p_rv_lv_ratio_gte_1,
        p_rv_lv_ratio_lt_1,
    ])

    return exam_level

def get_exam_names(exam_names):
    target_cols = [
        'negative_exam_for_pe', 
        'indeterminate',
        'chronic_pe', 'acute_and_chronic_pe',           # not indeterminate. Only One is true.
        'central_pe', 'leftsided_pe', 'rightsided_pe',  # not indeterminate. At least One is true.
        'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',        # not indeterminate. Only One is true.
    ]

    new_exam_names = []
    for e in exam_names:
        for col in target_cols:
            new_exam_names.append(e + '_' + col)
    
    return new_exam_names

def check_consistency(sub, test_csv):
    str_split = sub.id.str.split('_', 1, expand=True)
    str_split.columns = ['StudyInstanceUID', 'label_type']
    
    condition = ~str_split.label_type.isnull()
    new_df = pd.concat([sub[condition], str_split[condition]], axis=1)
    del new_df['id']
    df_exam = new_df.pivot(index='StudyInstanceUID', columns='label_type', values='label')
    
    condition = str_split.label_type.isnull()
    df_image = sub[condition]
    df_image = df_image.merge(test_csv, how='left', left_on='id', right_on='SOPInstanceUID')
    df_image.rename(columns = {"label": "pe_present_on_image"}, inplace=True)
    del df_image['id']
    
    df = df_exam.merge(df_image, how='left', on='StudyInstanceUID')
    ids = ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"]
    labels = [c for c in df.columns if c not in ids]
    df = df[ids + labels]

    # SPLIT NEGATIVE AND POSITIVE EXAMS
    df['positive_images_in_exam'] = df['StudyInstanceUID'].map(df.groupby(['StudyInstanceUID']).pe_present_on_image.max())

    df_pos = df.loc[df.positive_images_in_exam >  0.5]
    df_neg = df.loc[df.positive_images_in_exam <= 0.5]

    
    # CHECKING CONSISTENCY OF POSITIVE EXAM LABELS
    rule1a = df_pos.loc[((df_pos.rv_lv_ratio_lt_1  >  0.5)  & 
                         (df_pos.rv_lv_ratio_gte_1 >  0.5)) | 
                        ((df_pos.rv_lv_ratio_lt_1  <= 0.5)  & 
                         (df_pos.rv_lv_ratio_gte_1 <= 0.5))].reset_index(drop = True)
    rule1a['broken_rule'] = '1a'

    rule1b = df_pos.loc[(df_pos.central_pe    <= 0.5) & 
                        (df_pos.rightsided_pe <= 0.5) & 
                        (df_pos.leftsided_pe  <= 0.5)].reset_index(drop = True)
    rule1b['broken_rule'] = '1b'

    rule1c = df_pos.loc[(df_pos.acute_and_chronic_pe > 0.5) & 
                        (df_pos.chronic_pe           > 0.5)].reset_index(drop = True)
    rule1c['broken_rule'] = '1c'

    rule1d = df_pos.loc[(df_pos.indeterminate        > 0.5) | 
                        (df_pos.negative_exam_for_pe > 0.5)].reset_index(drop = True)
    rule1d['broken_rule'] = '1d'
    
    
    # CHECKING CONSISTENCY OF NEGATIVE EXAM LABELS
    rule2a = df_neg.loc[((df_neg.indeterminate        >  0.5)  & 
                         (df_neg.negative_exam_for_pe >  0.5)) | 
                        ((df_neg.indeterminate        <= 0.5)  & 
                         (df_neg.negative_exam_for_pe <= 0.5))].reset_index(drop = True)
    rule2a['broken_rule'] = '2a'

    rule2b = df_neg.loc[(df_neg.rv_lv_ratio_lt_1     > 0.5) | 
                        (df_neg.rv_lv_ratio_gte_1    > 0.5) |
                        (df_neg.central_pe           > 0.5) | 
                        (df_neg.rightsided_pe        > 0.5) | 
                        (df_neg.leftsided_pe         > 0.5) |
                        (df_neg.acute_and_chronic_pe > 0.5) | 
                        (df_neg.chronic_pe           > 0.5)].reset_index(drop = True)
    rule2b['broken_rule'] = '2b'
    
    # MERGING INCONSISTENT PREDICTIONS
    errors = pd.concat([rule1a, rule1b, rule1c, rule1d, rule2a, rule2b], axis = 0)
    
    
    if len(errors) == 0:
        return True
    else:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',
                        '-c',
                        type=str,
                        required=True,
                        help='path of the config file')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_file)

    inference(cfg)


if __name__ == '__main__':
    main()