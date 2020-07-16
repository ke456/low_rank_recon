# Class for miscellaneous utilities
import numpy as np
import torch
import random

def fix_seeds(
    seed=90,
    set_system=True,
    set_torch=True,
    set_torch_cudnn=True):
    """Fix seeds for reproducibility.
        Parameters
        ----------
        seed : int
            Random seed to be set.
        set_system : bool
            Whether to set `np.random.seed(seed)` and `random.seed(seed)`
        set_torch : bool
            Whether to set `torch.manual_seed(seed)`
        set_torch_cudnn: bool
            Flag for whether to enable cudnn deterministic mode.
            Note that deterministic mode can have a performance impact.
            https://pytorch.org/docs/stable/notes/randomness.html
        """
    # set system seed
    if set_system:
        np.random.seed(seed)
        random.seed(seed)

    # set torch seed
    if set_torch:
        torch.manual_seed(seed)

    # set torch cudnn backend
    if set_torch_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

