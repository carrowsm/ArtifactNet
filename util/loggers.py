from pytorch_lightning.loggers import LightningLoggerBase, rank_zero_only
from pytorch_lightning.logging import TensorBoardLogger
import os
import json
import matplotlib.pyplot as plt



class MyLogger(LightningLoggerBase):
    def __init__(self, logdir=None, save_hparams=False, use_tb=True):

        if logdir == None :
            self.logdir = os.getcwd()
        else :
            self.logdir = logdir




    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        if save_hparams :
            d = var(params)
            hparam_file = os.path.join(self.logdir, "hparams.json")
            with open(hparam_file, "w") as f :
                json.dump(d, f)
        else :
            return

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        pass

    def save(self):
        # Optional. Any code necessary to save logger data goes here
        pass

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass



class TensorBoardCustom(TensorBoardLogger):
    """Custom changes to PyTorch Lightning's TensorBoardLogger."""
    def __init__(self, save_dir, name='default', save_hparams=False):
        super(TensorBoardCustom, self).__init__(save_dir, name=name, version=None)
        # self.logdir = save_dir
        self.save_hparams = save_hparams
        # self.name = name


    @rank_zero_only
    def log_hyperparams(self, params):
        # Bypass this as it does not currently work
        if self.save_hparams :
            return
        else :
            return

    def add_mpl_img(self, tag, X, step) :
        """ Creates a matplotlib image out of a 4D tensor or list of
        2D tensors:
        X.shape == (N, 1, D, H, W) or
        X       == [x_real, y_real, x_fake, y_fake] """
        cm = ['viridis', "Greys"]

        if isinstance(X, list) :
            # X is a list of 2D tensors
            N = len(X)
            if N == 4 :
                # Expect it to have a particular format
                titles = ["DA+ Real", "DA- Real", "DA+ Fake", "DA- Fake"]
            else :
                titles = np.arange(0, N, 1).astype(str)
            fig, ax = plt.subplots(nrows=1, ncols=N, figsize=[6*N, 6])
            for i in range(N) :
                img = ax[i].imshow(X[i], cmap=cm[1])
                ax[i].set_title(titles[i])
                fig.colorbar(img, ax=ax[i], shrink=0.6)

        else :
            # Currently dont support other image types
            return
        # Create tensorboard figure and add it
        self.experiment.add_figure(tag, fig, step)
