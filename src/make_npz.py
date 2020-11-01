from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from datasets import RSNADataset
from utils.general_utils import omegaconf_to_yaml

def make_npz(save_path, 
             csv_path, 
             img_dir, 
             file_extension,
             target_cols,
             image_size):
    
    df = pd.read_csv(csv_path)
    df["file_name"] = df.SOPInstanceUID + '.' + file_extension
    df.z_pos_order = df.z_pos_order.map(lambda x: f'{x:04}')
    df.file_name = df.z_pos_order + '_' + df.file_name
    df["path_to_series_id"] = str(img_dir) + '/' + \
        df.StudyInstanceUID + '/' + df.SeriesInstanceUID
    path_to_series_id = df["path_to_series_id"].unique()
    map_filename_to_index = {file_name: index
        for index, file_name in enumerate(df.file_name.values)}

    resize = A.Compose([
        A.Resize(image_size, image_size, p=1.0)])
    
    for index in tqdm(range(len(path_to_series_id))):
        series_path = Path(path_to_series_id[index])
        data_path_list = series_path.glob(f'*.{file_extension}')
        data_path_list = sorted(data_path_list)

        img_list, label_list = [], []
        for data_path in data_path_list:
            data = df.iloc[map_filename_to_index[data_path.name]]
            assert data_path.name == data.file_name
            img = cv2.imread(str(data_path))[..., ::-1]
            img = resize(image=img)['image']
            labels = data[target_cols].values.astype('float32')
            img_list.append(img)
            label_list.append(labels)

        img_list = np.stack(img_list)
        label_list = np.stack(label_list)
        
        img_list = img_list.transpose(0, 3, 1, 2) # (sequence, channel, h, w)

        study_id = series_path.parent.stem
        series_id = series_path.stem
        path = Path(save_path) / study_id / Path(series_id).with_suffix('.npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, imgs=img_list, labels=label_list)


if __name__ == '__main__':
    image_size = 512
    save_path = 'data/npz/train-jpegs-512'
    csv_path = 'data/fold/ver1/train-jpegs-512/train.csv'
    img_dir = 'data/train-jpegs-512'
    file_extension = 'jpg'
    target_cols = [
        'negative_exam_for_pe', 
        'indeterminate',
        'chronic_pe', 'acute_and_chronic_pe',           # not indeterminate. Only One is true.
        'central_pe', 'leftsided_pe', 'rightsided_pe',  # not indeterminate. At least One is true.
        'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',        # not indeterminate. Only One is true.
        'pe_present_on_image',
    ]

    make_npz(save_path,
            csv_path,
            img_dir,
            file_extension,
            target_cols,
            image_size)