import argparse
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
import yaml
import cv2
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

import models
from utils.general_utils import omegaconf_to_yaml, seed_everything


class Data(Dataset):
    def __init__(
        self,
        csv_path: str,
        img_dir: str,
    ):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_path)
        self.df["path_to_series_id"] = str(img_dir) + '/' + \
            self.df.StudyInstanceUID + '/' + self.df.SeriesInstanceUID
        self.path_to_series_id = self.df["path_to_series_id"].unique()

    def __len__(self):
        return len(self.path_to_series_id)
    
    def __getitem__(self, index):
        series_path = Path(self.path_to_series_id[index]).with_suffix('.npz')
        data = np.load(series_path)
        x = data['imgs'].astype('float32')
        labels = data['labels'].astype('float32')
        
        return x, labels


def dump_embeddings(save_path, 
                    csv_path, 
                    img_dir, 
                    ckpt_path,
                    n_workers,
                    gpu,
                    image_size,
                    chunk_size):

    net = prepare_model(ckpt_path, gpu)

    dataset = Data(csv_path, img_dir)
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=False,)

    resize = A.Compose([
        A.Resize(image_size, image_size, p=1.0)])
    
    for index, (x, labels) in enumerate(tqdm(loader)):
        b, s, c, h, w = x.size()
        x = x.view(s, c, h, w)

        if image_size != h and image_size != w:
            x = x.numpy().transpose(0, 2, 3, 1)
            x = [resize(image=img)['image'].transpose(2, 0, 1) for img in x]
            x = torch.tensor(np.stack(x))

        b, s, n = labels.size()
        labels = labels.view(s, n)

        embeddings = []
        for i in range(0, len(x), chunk_size):
            # Note: shape
            # embedding -> (chunk_size, n_embeddings)
            embedding = net.forward_until_pooling(
                x[i:i + chunk_size].to(gpu))
            embedding = embedding.detach().cpu()
            embeddings.extend(embedding.numpy())
        embeddings = np.array(embeddings)
        labels = labels.detach().cpu().numpy()

        study_id = Path(dataset.path_to_series_id[index]).parent.stem
        series_id = Path(dataset.path_to_series_id[index]).stem
        path = Path(save_path) / study_id / Path(series_id).with_suffix('.npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, embeddings=embeddings, labels=labels)

    with open(save_path / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(omegaconf_to_yaml(cfg), f)


def prepare_model(ckpt_path, gpu):
    ckpt_path = Path(ckpt_path)
    cfg_path = ckpt_path.parents[1] / 'train_config.yaml'
    cfg = OmegaConf.load(cfg_path)
    runName = cfg.experiment["runName"]

    print(f' runName: {runName}',
          f' ckpt_name: {ckpt_path.name}',
          sep='\n')

    net = getattr(models, cfg.model.name)(**cfg.model.args)
    net.to(gpu)
    ckpt = torch.load(str(ckpt_path), map_location=f'cuda:{gpu}')['state_dict']
    ckpt = {k[k.find('.') + 1:]: v for k, v in ckpt.items()}
    net.load_state_dict(ckpt, strict=True)
    net.eval()

    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',
                        '-c',
                        type=str,
                        required=True,
                        help='path of the config file')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_file)

    save_path = Path(cfg['save_root']) / \
        cfg['version'] / cfg['model'] / f'fold_{cfg["fold"]}'

    if save_path.exists():
        print('This version already exists.\n'
              f'version:{save_path}')
        ans = None
        while ans not in ['y', 'Y']:
            ans = input('Do you want to continue inference? (y/n): ')
            if ans in ['n', 'N']:
                quit()

    save_path.mkdir(exist_ok=True, parents=True)

    dump_embeddings(save_path,
                    cfg.csv_path,
                    cfg.img_dir,
                    cfg.ckpt_path,
                    cfg.n_workers,
                    cfg.gpu,
                    cfg.image_size,
                    cfg.chunk_size,)
