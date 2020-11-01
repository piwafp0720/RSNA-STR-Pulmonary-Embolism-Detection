from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import losses
import models
from utils.general_utils import omegaconf_to_yaml, seed_everything


def train(cfg,
          dataset,
          dataset_args,
          module,
          ckpt_name='{epoch:02}-{val_loss:.3f}',
          kwargs={}):

    gpu = cfg.experiment.gpu
    torch.cuda.set_device(gpu)
    seed = np.random.randint(
        65535) if cfg.experiment.seed is None else cfg.experiment.seed
    seed_everything(seed)

    logger = WandbLogger(
        name=cfg.experiment.runName,
        project=cfg.experiment.project,
        offline=cfg.experiment.offline)
    logger.experiment.config.update(dict(cfg))
    logger.experiment.config.update({"dir": logger.experiment.dir})

    ckpt_dir = Path(logger.experiment.dir) / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    checkpoint = ModelCheckpoint(
        filepath=ckpt_dir / ckpt_name,
        save_top_k=cfg.checkpoint.save_top_k,
        save_weights_only=cfg.checkpoint.save_weights_only,
        verbose=True,
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode)

    trainer = Trainer(
        logger=logger,
        max_epochs=cfg.train.epoch,
        accumulate_grad_batches=cfg.train.n_accumulations,
        limit_val_batches=1.0,
        val_check_interval=cfg.train.val_check_interval,
        early_stop_callback=cfg.train.early_stopping,
        gpus=[gpu],
        checkpoint_callback=checkpoint,
        precision=16 if cfg.train.amp else 32,
        amp_level=cfg.train.amp_level,
    )

    net = getattr(models, cfg.model.name)(**cfg.model.args)
    net.to(gpu)

    if cfg.model.load_checkpoint is not None:
        ckpt = torch.load(cfg.model.load_checkpoint,
                          map_location=f'cuda:{gpu}')['state_dict']
        ckpt = {k[k.find('.') + 1:]: v for k, v in ckpt.items()}
        net.load_state_dict(ckpt, strict=False)
        print(f'\nload checkpoint: {cfg.model.load_checkpoint}\n')

    loss_args = dict(cfg.loss.args) if cfg.loss.args else {}
    loss = getattr(losses, cfg.loss.name)(**loss_args).cuda(gpu)

    if cfg.optimizer.scheduler.name == 'CosineAnnealingLR':
        cfg.optimizer.scheduler.args.T_max = cfg.train.epoch

    model = module(dataset,
                   dataset_args,
                   cfg.train.batch_size,
                   net,
                   loss,
                   n_workers=cfg.experiment.n_workers,
                   optimizer=cfg.optimizer.name,
                   optimizer_args=cfg.optimizer.args,
                   scheduler=cfg.optimizer.scheduler.name,
                   scheduler_args=cfg.optimizer.scheduler.args,
                   freeze_start=cfg.model.freeze_start.target_epoch,
                   unfreeze_params=cfg.model.freeze_start.unfreeze_params,
                   **kwargs)

    with open(ckpt_dir.parent / 'train_config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(omegaconf_to_yaml(cfg), f)
    
    with open(ckpt_dir.parent / 'augmentation.txt', 'w', encoding='utf-8') as f:
        transform_train = dataset_args['train']['transform']
        transform_val = dataset_args['val']['transform']
        f.write("---train augmentation---\n")
        if transform_train is not None:
            f.write(str(transform_train.transform) + "\n\n")
        f.write("---val augmentation---\n")
        if transform_val is not None:
            f.write(str(transform_val.transform) + "\n")

    trainer.fit(model)

    for n, model_path in enumerate(Path(logger.experiment.dir).glob('**/*.ckpt')):
        name = f"model_{n}"
        logger.experiment.config.update({name: model_path})
    
    if cfg.experiment.network_type == 'rnn':
        cnn_path = OmegaConf.load(
            Path(cfg.dataset.img_dir) / 'config.yaml').ckpt_path
        logger.experiment.config.update({'cnn_path': cnn_path})
