import os
import json
import torch
import logging
import argparse
from model.model import get_model_instance
from model.loss import get_loss_function
from model.metric import get_metric_functions
from data_loaders import get_dataloader_instance
from logger import Logger
from trainer import Trainer
from utils.util import log_model_summary

logging.basicConfig(level=logging.INFO, format='')


def main(config, resume, experiment_path):
    train_logger = Logger(experiment_path)

    data_loader = get_dataloader_instance(config['data_loader']['name'], config)
    valid_data_loader = data_loader.get_validation_loader()

    model = get_model_instance(config['model_name'], **config['model_params'])
    log_model_summary(model)

    loss = get_loss_function(config['loss'])
    metrics = get_metric_functions(config['metrics'])

    trainer = Trainer(model, loss, metrics,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      train_logger=train_logger)

    trainer.train()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Template')
    arg_group = parser.add_mutually_exclusive_group(required=True)
    arg_group.add_argument('-c', '--config', default=None, type=str,
                           help='config file path (default: None)')
    arg_group.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']
        experiment_path = args.resume
    else:
        config = json.load(open(args.config))
        experiment_path = os.path.join(config['trainer']['save_dir'], config['experiment_name'])
        assert not os.path.exists(experiment_path), "Path {} already exists!".format(experiment_path)

    main(config, args.resume, experiment_path)
