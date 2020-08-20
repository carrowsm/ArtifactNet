import os
import json
import numpy as np
import matplotlib.pyplot as plt

from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.logging import TensorBoardLogger





class TensorBoardCustom(TensorBoardLogger):
    """Custom changes to PyTorch Lightning's TensorBoardLogger."""
    def __init__(self, save_dir, name='default', save_hparams=False):
        super(TensorBoardCustom, self).__init__(save_dir, name=name, version=None)
        self.save_hparams = save_hparams


    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # Choose which metrics to plot
        trg_metrics, val_metrics = {}, {}
        for k, v in metrics.items() :
            if k not in ['epoch'] :
                # Don't log epoch number
                if "val" in k :
                    d = val_metrics
                else :
                    d = trg_metrics

                if isinstance(v, dict) :
                    # If there is a dictionary with more loss values, plot each one
                    # for this to work, the key should have format d_loss or g_loss
                    for k_i, v_i in v.items() :
                        d[k_i] = v_i
                else :
                    d[k] = v

        # Add both training and validation metrics to logger
        self.experiment.add_scalars("trg_loss/", trg_metrics, step)
        self.experiment.add_scalars("val_loss/", val_metrics, step)


    def add_mpl_img(self, tag, X, step, clip_vals=False) :
        """ Creates a matplotlib image out of a 4D tensor or list of
        2D tensors:
        X.shape == (N, 1, D, H, W) or
        X       == [x_real, x_fake, y_real, y_fake] """
        cm = ['viridis', "Greys"]

        if isinstance(X, list) :
            # X is a list of 2D tensors
            N = len(X)
            if N == 4 :
                # Expect it to have a particular format
                titles = ["DA+ Real", "DA+ Fake", "DA- Real", "DA- Fake"]
            else :
                titles = np.arange(0, N, 1).astype(str)
            fig, ax = plt.subplots(nrows=1, ncols=N, figsize=[6*N, 6])
            for i in range(N) :
                if clip_vals :
                    X[i] = np.clip(X[i], -1.0, 1.0)
                img = ax[i].imshow(X[i], cmap=cm[1])
                ax[i].set_title(titles[i])
                fig.colorbar(img, ax=ax[i], shrink=0.6)

        else :
            # Currently dont support other image types
            return
        # Create tensorboard figure and add it
        self.experiment.add_figure(tag, fig, step)
