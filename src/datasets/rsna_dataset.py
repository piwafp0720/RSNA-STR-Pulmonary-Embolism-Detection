from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import pydicom
from torch.utils.data import Dataset
from turbojpeg import TurboJPEG


class RSNADataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        file_extension: str = 'dcm',
        mode: str = 'train',
        fold: int = 0,
        k_fold: int = 5,
        transform=None,
        network_type: str = 'cnn',
        max_sequence: int = 1083,
    ):
        assert mode in ['train', 'val', 'test']
        assert network_type in ['cnn', 'rnn', 'cnn_rnn']
        assert -1 <= fold < 5
        assert 15 % k_fold == 0

        self.transform = transform
        self.csv_path = Path(csv_path)
        self.img_dir = Path(img_dir)
        self.file_extension = file_extension
        self.mode = mode
        self.network_type = network_type
        self.max_sequence = max_sequence

        if network_type == 'cnn':
            self.target_cols = [
                'pe_present_on_image',
            ]
        elif network_type == 'rnn':
            self.target_cols = [
                'negative_exam_for_pe', 
                'indeterminate',
                'chronic_pe', 'acute_and_chronic_pe',           # not indeterminate. Only One is true.
                'central_pe', 'leftsided_pe', 'rightsided_pe',  # not indeterminate. At least One is true.
                'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',        # not indeterminate. Only One is true.
                'pe_present_on_image',
            ]
        
        if self.file_extension == 'jpg':
            self.jpeg_reader = TurboJPEG()

        df = pd.read_csv(self.csv_path)
        df["file_name"] = df.SOPInstanceUID + '.' + self.file_extension
        if self.file_extension != 'dcm':
            df.z_pos_order = df.z_pos_order.map(lambda x: f'{x:04}')
            df.file_name = df.z_pos_order + '_' + df.file_name
        df["image_name"] = str(self.img_dir) + '/' + \
            df.StudyInstanceUID + '/' +  df.SeriesInstanceUID + '/' +  df.file_name
        self.df = df if fold == -1 else self._make_fold(df, fold, k_fold, mode=mode)

        if self.network_type == 'rnn' or self.network_type == 'cnn_rnn':
            self.df["path_to_series_id"] = str(self.img_dir) + '/' + \
                self.df.StudyInstanceUID + '/' + self.df.SeriesInstanceUID
            self.path_to_series_id = self.df["path_to_series_id"].unique()
        
    def __len__(self):
        if self.network_type == 'cnn':
            return len(self.df)
        elif self.network_type == 'rnn':
            return len(self.path_to_series_id)
        elif self.network_type == 'cnn_rnn':
            return len(self.path_to_series_id)

    def __getitem__(self, index):
        if self.network_type == 'cnn':
            return self._get_single_image(index)
        else:
            return self._get_series(index)
    
    def _get_single_image(self, index):
        data = self.df.iloc[index]

        return self._get_img_label(data)
    
    def _get_series(self, index):
        if self.network_type == 'rnn':
            return self._read_embeddings(index)
        elif self.network_type == 'cnn_rnn':
            if self.file_extension == 'npz':
                return self._read_series_npz(index)
            else:
                return self._read_series_images(index)
    
    def _read_embeddings(self, index):
        data_path = self.path_to_series_id[index]
        data_path = Path(data_path).with_suffix('.npz')
        data = np.load(data_path)
        embeddings = data['embeddings']
        labels = data['labels']
        sequence_length, _ = embeddings.shape
        
        embeddings = self._padding_sequence(sequence_length, embeddings, 0)
        labels = self._padding_sequence(sequence_length, labels, -1)
        
        return embeddings, labels, sequence_length

    def _read_series_npz(self, index):
        data_path = Path(self.path_to_series_id[index]).with_suffix('.npz')
        data = np.load(data_path)
        imgs = data['imgs'] #(sequence, 3, h, w)
        labels = data['labels'] #(sequence, n_class)
        if self.transform is not None:
            imgs = imgs.transpose(0, 2, 3, 1)
            imgs = [self.transform(image=img).transpose( 
                2, 0, 1) for img in imgs]
            imgs = np.stack(imgs)

        imgs = imgs.astype('float32')
        labels = labels.astype('float32')

        return imgs, labels

    def _read_series_images(self, index):
        # use only when inference.
        data_path = self.path_to_series_id[index]
        dicoms, dicom_files = self._load_dicom_array(data_path)
        imgs = self._get_three_windowing_image(dicoms)
        if self.transform is not None:
            imgs = imgs.transpose(0, 2, 3, 1)
            imgs = [self.transform(image=img).transpose( 
                2, 0, 1) for img in imgs]
            imgs = np.stack(imgs)

        imgs = imgs.astype('float32')

        exam_level_name, image_level_name = self._get_file_names(dicom_files)
        
        return imgs, exam_level_name, image_level_name

    def _get_img_label(self, data):
        if self.file_extension == 'jpg':
            binary = open(data.image_name, "rb")
            img = self.jpeg_reader.decode(binary.read(), 0)
        elif self.file_extension == 'dcm':
            raise NotImplementedError
        if self.transform is not None:
            img = self.transform(image=img)
        img = img.transpose(2, 0, 1).astype('float32')

        labels = data[self.target_cols].values.astype('float32')

        return img, labels

    def _make_fold(self, df, fold, k_fold, mode='train'):
        df_new = df.copy()
        offset = 15 // k_fold
        target = [i + fold * offset for i in range(offset)]

        if mode == 'train':
            df_new = df_new.query(f'fold not in {target}')
        else:
            df_new = df_new.query(f'fold in {target}')
        
        return df_new
    
    def _padding_sequence(self, sequence_length, target, value):
        pad_len = self.max_sequence - sequence_length
        assert pad_len >= 0
            
        if pad_len > 0:
            padding = [np.full_like(target[0], value)] * pad_len
            target = np.concatenate([target, padding])
        
        return target

    def _load_dicom_array(self, path_to_series_id):
        dicom_files = list(Path(path_to_series_id).glob('*.dcm'))
        dicoms = [pydicom.dcmread(d) for d in dicom_files]
        slope = float(dicoms[0].RescaleSlope)
        intercept = float(dicoms[0].RescaleIntercept)
        # Assume all images are axial
        z_pos = [float(d.ImagePositionPatient[-1]) for d in dicoms]
        dicoms = np.asarray([d.pixel_array for d in dicoms])
        dicoms = dicoms[np.argsort(z_pos)]
        dicoms = dicoms * slope
        dicoms = dicoms + intercept

        dicom_files = np.array(dicom_files)[np.argsort(z_pos)]

        return dicoms, dicom_files
    
    def _windowing(self, img, window_length, window_width):
        upper = window_length + window_width // 2
        lower = window_length - window_width // 2
        x = np.clip(img.copy(), lower, upper)
        x = x - np.min(x)
        x = x / np.max(x)
        x = (x * 255.0).astype('uint8')

        return x
    
    def _get_three_windowing_image(self, dicoms):
        img_lung = np.expand_dims(
            self._windowing(dicoms, -600, 1500), axis=1)
        img_mediastinal = np.expand_dims(
            self._windowing(dicoms, 40, 400), axis=1)
        img_pe_specific = np.expand_dims(
            self._windowing(dicoms, 100, 700), axis=1)
        
        return np.concatenate([
            img_lung, img_pe_specific, img_mediastinal], axis=1)
    
    def _get_file_names(self, dicom_files):
        exam_level_name = str(dicom_files[0].parent.parent.stem)
        dicom_files = dicom_files.tolist()
        image_level_name = list(map(lambda x: str(x.stem), dicom_files))
        
        return exam_level_name, image_level_name


if __name__ == '__main__':
    # csv_path = '/raid/others/kaggle_rsna_2020/data/fold/ver1/train.csv'
    # csv_path = '/raid/others/kaggle_rsna_2020/data/fold/ver1/train_mini.csv'
    csv_path = '/raid/others/kaggle_rsna_2020/data/test.csv'

    # network_type = 'cnn'
    # img_dir = '/raid/others/kaggle_rsna_2020/data/train-jpegs-256'
    # file_extension = 'jpg'
    # network_type = 'rnn'
    # img_dir = '/raid/others/kaggle_rsna_2020/dump/baseline_1'
    # file_extension = 'npz'
    network_type = 'cnn_rnn'
    # img_dir = '/raid/others/kaggle_rsna_2020/data/npz/train-jpegs-256'
    # file_extension = 'npz'
    # mode = 'train'
    img_dir = '/raid/others/kaggle_rsna_2020/data/test'
    file_extension = 'dcm'
    mode = 'test'
    fold = -1
    k_fold = 5

    dataset = RSNADataset(csv_path,
                          img_dir,
                          file_extension=file_extension,
                          mode=mode,
                          fold=fold,
                          k_fold=k_fold,
                          network_type=network_type,
                          )

    if network_type == 'cnn':
        n = np.random.randint(len(dataset))
        import time
        start = time.time()
        img, labels = dataset[n]
        print(time.time() - start)

        print(n, len(dataset))
        print(img.shape, labels)
    else:
        n = np.random.randint(len(dataset))
        import time
        start = time.time()
        if network_type == 'rnn':
            img_list, label_list, sequence_length = dataset[n]
        elif network_type == 'cnn_rnn':
            if mode == 'test':
                img_list = dataset[n]
            else:
                img_list, label_list = dataset[n]

        print(time.time() - start)
        print(n, len(dataset))
        if mode == 'test':
            print(img_list.shape)
        else:
            print(img_list.shape, label_list.shape)