import torch
import numpy as np
from base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, test_data_loader=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.batch_size = data_loader.batch_size
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(self.batch_size))

    def _to_tensor(self, data, target):
        """
         Handle GPU is applicable
        Args:
            data: batch of training data
            target: batch of labels

        Returns:
            data, target
        """
        if self.with_cuda:
            data, target = data.to(self.gpu), target.to(self.gpu)
        return data, target

    def _eval_metrics(self, output, target):
        """
        Evaluate metrics
        Args:
            output: model output
            target: target labels

        Returns:
            list of metrics
        """
        acc_metrics = np.zeros(len(self.metrics))
        output = output.cpu().data.numpy()
        target = target.cpu().data.numpy()
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        Args:
            epoch: current training epoch number

        Returns:
            log with training info
        """
        self.model.train()

        losses = []
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = self._to_tensor(data, target)
            self.optimizer.zero_grad()
            output = self.model(data)

            loss = self.loss(output, target)
            loss.backward()

            self.optimizer.step()
            losses.append(loss.data.mean())

            output = torch.sigmoid(output)
            accuracy_metrics = self._eval_metrics(output, target)
            total_metrics += accuracy_metrics

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} Accuracy: {:.6f}'.format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        len(self.data_loader) * self.data_loader.batch_size,
                        100.0 * batch_idx / len(self.data_loader),
                        loss.data.mean(),
                        accuracy_metrics[0] if len(accuracy_metrics) else 0
                    )
                )

        log = {
            'loss': float(np.mean(losses)),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch()
            log = {**log, **val_log}

        return log

    def _valid_epoch(self):
        """
        Validate after training an epoch

        Returns:
             а log that contains information about validation

        """
        self.model.eval()
        val_losses = []
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = self._to_tensor(data, target)

                output = self.model(data)
                loss = self.loss(output, target)

                val_losses.append(loss.data.mean())
                total_val_metrics += self._eval_metrics(output, target)

        return {
            'val_loss': float(np.mean(val_losses)),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

    def test(self):
        """
        Test logic.

        Test is performed on data without augmentations.

        """
        self.model.eval()
        test_losses = []
        total_test_metrics = np.zeros(len(self.metrics))
        false_positives = np.zeros(self.config['model_params']['num_classes'])
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_data_loader):
                data, target = self._to_tensor(data, target)

                output = self.model(data)

                loss = self.loss(output, target)

                test_losses.append(loss.data.mean())

                output = torch.sigmoid(output)
                # TODO make threshold configurable
                false_positives += (output.data.numpy() > 0.5).sum(axis=0)

                accuracy_metrics = self._eval_metrics(output, target)
                total_test_metrics += accuracy_metrics

                self.logger.info(
                    'Test: [{}/{} ({:.0f}%)] Loss: {:.6f} Accuracy: {:.6f}'.format(
                        batch_idx * self.test_data_loader.batch_size,
                        len(self.test_data_loader) * self.test_data_loader.batch_size,
                        100.0 * batch_idx / len(self.test_data_loader),
                        loss.data.mean(),
                        accuracy_metrics[0] if len(accuracy_metrics) else 0
                    )
                )

        self.logger.info("Total test Loss: {:.6f} Accuracy: {:.6f}".format(
            float(np.mean(test_losses)),
            (total_test_metrics / len(self.test_data_loader)).tolist()[0]
        ))
        self.logger.info("Test False Positives: {}".format(false_positives))
