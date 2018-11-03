import os
import urllib.request
import zipfile

import glog as log
import numpy as np


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def log_model_summary(model, verbose=False):
    if verbose:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        log.info('Trainable parameters: {}'.format(params))
    log.info(model)


def download_and_unzip(url, filename, dest_dir):
    fpath = os.path.join(dest_dir, filename)
    if os.path.isfile(fpath):
        log.info("Files already downloaded")
        return
    log.info("Downloading " + filename + " to " + dest_dir)
    os.makedirs(dest_dir)
    urllib.request.urlretrieve(url, fpath)
    with zipfile.ZipFile(fpath) as zf:
        zf.extractall(dest_dir)

def create_val_folder(data_dir):
    """
    This method is responsible for separating validation images into separate sub folders
    """
    path = os.path.join(data_dir, 'val/images')  # path where validation data is present now
    filename = os.path.join(data_dir, 'val/val_annotations.txt')  # file where image2class mapping is present
    fp = open(filename, "r")  # open file in read mode
    data = fp.readlines()  # read line by line

    # Create a dictionary with image names as key and corresponding classes as values
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path, folder))
        if not os.path.exists(newpath):  # check if folder exists
            os.makedirs(newpath)

        if os.path.exists(os.path.join(path, img)):  # Check if image exists in default directory
            os.rename(os.path.join(path, img), os.path.join(newpath, img))


class EarlyStopping:
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            log.info('Stopping early. Invalid metrics')
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            log.info('Stopping early. Model did not improve in {} epochs'.format(self.patience))
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta
