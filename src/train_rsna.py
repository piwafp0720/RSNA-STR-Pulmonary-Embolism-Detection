import argparse

import torch
from omegaconf import OmegaConf

from augmentations import RSNAAugmentation
from datasets import RSNADataset
from modules import RSNAModule
from train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',
                        '-c',
                        type=str,
                        required=True,
                        help='path of the config file')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_file)

    dataset = RSNADataset
    if cfg.experiment.network_type == 'rnn':
        transform_train = None
        transform_val = None
    else:
        transform_train = RSNAAugmentation(mode='train', **cfg.transform)
        transform_val = RSNAAugmentation(mode='val', **cfg.transform)

    dataset_args = {
        'train': {
            'transform': transform_train,
            'mode': 'train',
            'network_type': cfg.experiment.network_type,
            **cfg.dataset
        },
        'val': {
            'transform': transform_val,
            'mode': 'val',
            'network_type': cfg.experiment.network_type,
            **cfg.dataset
        }
    }

    rsna_module_kwargs = {}
    network_type = cfg.experiment.network_type
    rsna_module_kwargs['network_type'] = network_type

    if network_type == 'rnn' or network_type == 'cnn_rnn':
        rsna_module_kwargs['n_classes'] = cfg.model.args.n_classes

    module = RSNAModule

    ckpt_name = '{epoch:02}-{val_loss:.3f}'

    train(cfg, dataset, dataset_args, module, ckpt_name=ckpt_name, kwargs=rsna_module_kwargs)


if __name__ == '__main__':
    main()
