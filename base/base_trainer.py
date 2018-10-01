import os
import math
import torch
import glog as log
import torch.optim as optim


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, loss, metrics, resume, config, experiment_path, train_logger=None):

        # training params
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.config = config
        self.train_logger = train_logger
        self.experiment_name = config['experiment_name']
        self.epochs = config['trainer']['epochs']
        self.save_freq = config['trainer']['save_freq']
        self.verbosity = config['trainer']['verbosity']
        self.optimizer = getattr(optim, config['optimizer_type'])(model.parameters(), **config['optimizer_params'])

        self.logger = log
        # handle GPU
        self.with_cuda = config['cuda'] and torch.cuda.is_available()
        if config['cuda'] and not torch.cuda.is_available():
            self.logger.warning('Warning: There\'s no CUDA support on this machine, '
                                'training is performed on CPU.')
        else:
            self.gpu = torch.device('cuda:' + str(config['gpu']))
            self.model = self.model.to(self.gpu)

        # lr scheduler params
        self.lr_scheduler = None
        if config['lr_scheduler']:
            self.lr_scheduler = getattr(
                optim.lr_scheduler,
                config['lr_scheduler']['lr_scheduler_type'], None
            )
            if self.lr_scheduler:
                self.lr_scheduler = self.lr_scheduler(self.optimizer, **config['lr_scheduler']['additional_params'])
                self.lr_scheduler_freq = config['lr_scheduler']['lr_scheduler_freq']

        # monitor metrics params
        self.monitor = config['trainer']['monitor']
        self.monitor_mode = config['trainer']['monitor_mode']
        assert self.monitor_mode == 'min' or self.monitor_mode == 'max'
        self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf

        self.start_epoch = 1
        self.checkpoint_dir = experiment_path
        if resume:
            self._resume_checkpoint(resume)

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # log training info
            log = {'epoch': epoch}

            for key, value in result.items():
                if key == 'metrics':
                    for i, metric in enumerate(self.metrics):
                        log[metric.__name__] = result['metrics'][i]
                elif key == 'val_metrics':
                    for i, metric in enumerate(self.metrics):
                        log['val_' + metric.__name__] = result['val_metrics'][i]
                else:
                    log[key] = value

            if self.train_logger is not None:
                self.train_logger.log_epoch(log)
                if self.verbosity >= 1:
                    for key, value in log.items():
                        self.logger.info('    {:15s}: {}'.format(str(key), value))

            # monitor metrics
            if (self.monitor_mode == 'min' and log[self.monitor] < self.monitor_best)\
                    or (self.monitor_mode == 'max' and log[self.monitor] > self.monitor_best):
                self.monitor_best = log[self.monitor]
                self._save_checkpoint(epoch, log, save_best=True)

            # save checkpoint
            if epoch % self.save_freq == 0:
                self._save_checkpoint(epoch, log)

            # lr_scheduler logic
            if self.lr_scheduler and epoch % self.lr_scheduler_freq == 0:
                self.lr_scheduler.step(epoch)
                lr = self.lr_scheduler.get_lr()[0]
                self.logger.info('New Learning Rate: {:.6f}'.format(lr))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        Args:
             epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, log, save_best=False):
        """
        Saving checkpoints

        Args:
            epoch: current epoch number
            log: logging information of the epoch
            save_best: if True, rename the experiments checkpoint to 'model_best.pth.tar'
        """
        model = type(self.model).__name__
        state = {
            'model': model,
            'epoch': epoch,
            'logger': self.train_logger,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best,
            'config': self.config
        }
        filename = os.path.join(
            self.checkpoint_dir,
            'checkpoint-epoch{:03d}-loss-{:.4f}.pth.tar'.format(epoch, log['loss'])
        )
        torch.save(state, filename)
        if save_best:
            os.rename(filename, os.path.join(self.checkpoint_dir, 'model_best.pth.tar'))
            self.logger.info("Saving current best: {} ...".format('model_best.pth.tar'))
        else:
            self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from experiments checkpoints

        Args:
            resume_path: checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.with_cuda:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(self.gpu)
        self.train_logger = checkpoint['logger']
        self.config = checkpoint['config']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))
