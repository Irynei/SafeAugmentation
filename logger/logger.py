import json
import os


class Logger:
    """
    Training process logger

    """
    def __init__(self, experiment_path):
        # TODO maybe add Tensorboard / Visdom logging
        self.entries = {}
        self.experiment_path = experiment_path
        self.log_epoch_file_path = os.path.join(experiment_path, 'log_epoch.txt')

    def log_epoch(self, entry):
        self.entries[len(self.entries) + 1] = entry
        with open(self.log_epoch_file_path, 'a') as file:
            file.write(json.dumps(entry) + '\n')

    def __str__(self):
        return json.dumps(self.entries, sort_keys=True, indent=4)
