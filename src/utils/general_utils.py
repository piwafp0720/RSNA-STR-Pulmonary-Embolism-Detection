import logging
import os
import random
import time
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
from omegaconf import dictconfig, listconfig


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None):
    t0 = time.time()
    yield
    msg = f'[{name}] done in {time.time()-t0:.2f} s'
    if logger:
        logger.info(msg)
    else:
        print(msg)


def omegaconf_to_yaml(cfg):
    if isinstance(cfg, dictconfig.DictConfig):
        cfg = dict(cfg)
        for key, value in cfg.items():
            cfg[key] = omegaconf_to_yaml(value)
    elif isinstance(cfg, listconfig.ListConfig):
        cfg = list(cfg)
        for value in cfg:
            value = omegaconf_to_yaml(value)
    else:
        return cfg
    return cfg
