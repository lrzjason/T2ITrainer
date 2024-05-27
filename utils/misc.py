import collections
import datetime
import os
import random
import subprocess
import time
from multiprocessing import JoinableQueue, Process

import numpy as np
import torch
import torch.distributed as dist
from mmcv import Config
from mmcv.runner import get_dist_info

from utils.dist_utils import get_rank
from utils.logger import get_root_logger

os.environ["MOX_SILENT_MODE"] = "1"  # mute moxing log


# from https://github.com/PixArt-alpha/PixArt-sigma (MIT) utils/misc.py
def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False