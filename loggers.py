from pytorch_lightning.loggers import LightningLoggerBase, rank_zero_only
import os
import json


def log_imgs(img1, img2) :
    

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
