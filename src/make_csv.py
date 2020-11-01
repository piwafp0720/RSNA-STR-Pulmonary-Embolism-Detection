import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
tqdm.pandas()

def add_z_pos(p_train_df, jpeg_path):
    train_csv = pd.read_csv(p_train_df)
    jpeg_path = Path(jpeg_path)
    path_list_jpg = list(jpeg_path.glob('**/*.jpg'))

    parsed = list(map(
        lambda x: str(x).split('.')[0].split('/')[-1], path_list_jpg))
    df_z_pos_order = pd.Series(parsed).str.split('_', expand=True)
    df_z_pos_order.columns = ['z_pos_order', 'SOPInstanceUID']

    order_dict = {
        key: cnt for cnt, key in enumerate(train_csv.SOPInstanceUID)}
    df_z_pos_order["order"] = df_z_pos_order[
        "SOPInstanceUID"].apply(lambda x: order_dict[x])
    df_z_pos_order = df_z_pos_order.sort_values('order').reset_index(drop=True)

    assert np.all(train_csv.SOPInstanceUID == df_z_pos_order.SOPInstanceUID)
    train_csv['z_pos_order'] = df_z_pos_order['z_pos_order']

    return train_csv

def split(train_csv):
    train_group_by = train_csv.groupby("StudyInstanceUID").sum().reset_index()
    train_group_by.negative_exam_for_pe = train_group_by.negative_exam_for_pe > 0
    target_cols = ['negative_exam_for_pe']

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for n, (train_index, val_index) in enumerate(skf.split(train_group_by,train_group_by[target_cols])):
        train_group_by.loc[val_index, 'fold'] = int(n)
    train_group_by['fold'] = train_group_by['fold'].astype(int)

    fold_dict = {}
    for key, value in zip(train_group_by.StudyInstanceUID, train_group_by.fold):
        fold_dict[key] = value

    train_csv['fold'] = train_csv['StudyInstanceUID'].apply(lambda x: fold_dict[x])

    return train_csv

def main(p_train_df, jpeg_path):
    print('get z pos ...')
    train_csv = add_z_pos(p_train_df, jpeg_path)
    print('split ...')
    train_csv = split(train_csv)

    p_output = Path('data/fold/ver1/train-jpegs-512')
    p_output.mkdir(exist_ok=True, parents=True)
    train_csv.to_csv(p_output / 'train.csv', index=False)

if __name__ == '__main__':
    p_train_df = 'data/train.csv'
    jpeg_path = 'data/train-jpegs-512'
    n_splits = 3 * 5
    main(p_train_df, jpeg_path)