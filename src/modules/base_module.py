from typing import Union

import optimizers
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset


class BaseModule(pl.LightningModule):
    def __init__(
        self,
        dataset: Dataset,
        dataset_args: dict,
        batch_size: int,
        model: nn.Module,
        loss: Union[nn.Module, dict],
        n_workers: int = 0,
        optimizer: Union[str, None] = None,
        optimizer_args: dict = {},
        scheduler: Union[str, None] = None,
        scheduler_args: dict = {},
        freeze_start: Union[int, None] = None,
        unfreeze_params: Union[list, None] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.dataset_args = dataset_args
        self.batch_size = batch_size
        self.model = model
        self.loss = loss
        self.n_workers = n_workers
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.scheduler = scheduler
        self.scheduler_args = scheduler_args
        self.freeze_start = freeze_start
        self.unfreeze_params = unfreeze_params

    def forward(self, *x):
        y = self.model(*x)
        return y

    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self.forward(x)

        loss = self.loss(y, t)
        logger_logs = {'loss': loss}

        output = {'loss': loss, 'progress_bar': {}, 'log': logger_logs}
        return output

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self.forward(x)

        val_batch_loss = self.loss(y, t)

        output = {
            'val_batch_loss': val_batch_loss,
        }
        return output

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack(
            [output['val_batch_loss'] for output in outputs]).mean()

        results = {
            'val_loss': val_loss,
            'progress_bar': {
                'val_loss': val_loss,
            },
            'log': {
                'val_loss': val_loss,
            }
        }
        return results

    def configure_optimizers(self):
        if OmegaConf.is_list(self.optimizer_args):
            kwargs_dict = {}
            for kwargs in self.optimizer_args:
                param_names = kwargs.pop('params')
                if param_names == 'default':
                    default_kwargs = kwargs
                else:
                    if isinstance(param_names, str):
                        param_names = [param_names]

                    for param in param_names:
                        kwargs_dict[param] = kwargs

            optimized_params = []
            for n, p in self.model.named_parameters():
                for i, (param, kwargs) in enumerate(kwargs_dict.items()):
                    if param in n:
                        optimized_params.append({'params': p, **kwargs})
                        break
                    elif i == len(kwargs_dict) - 1:
                        optimized_params.append({
                            'params': p,
                        })

            optimizer = getattr(optimizers, self.optimizer)(optimized_params,
                                                            **default_kwargs)

        elif OmegaConf.is_dict(self.optimizer_args):
            optimizer = getattr(optimizers,
                                self.optimizer)(self.parameters(),
                                                **self.optimizer_args)
        else:
            raise TypeError

        if self.scheduler is None:
            return optimizer
        else:
            scheduler = getattr(optimizers,
                                self.scheduler)(optimizer,
                                                **self.scheduler_args)
            return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = self.dataset(**self.dataset_args['train'])
        loader = DataLoader(dataset=dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.n_workers,
                            pin_memory=False)
        return loader

    def val_dataloader(self):
        dataset = self.dataset(**self.dataset_args['val'])
        loader = DataLoader(dataset=dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.n_workers,
                            pin_memory=False)
        return loader

    def on_epoch_start(self):
        if (self.freeze_start and self.unfreeze_params
                and self.current_epoch == 0):
            self.freeze()
            print('==Freeze Start==')
            print('Unfreeze params:')
            for n, p in self.model.named_parameters():
                if any(param in n for param in self.unfreeze_params):
                    p.requires_grad = True
                    print(f'  {n}')
            self.model.train()

        if self.current_epoch == self.freeze_start:
            print("==Unfreeze==")
            self.unfreeze()

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            v_num = tqdm_dict.pop('v_num')
        tqdm_dict['v_num'] = v_num[-7:]
        return tqdm_dict
