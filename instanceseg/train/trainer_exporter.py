import os
import os.path as osp
import datetime
import pytz


MY_TIMEZONE = 'America/New_York'


class TrainerExporter(object):
    log_headers = [
        'epoch',
        'iteration',
        'train/losses',
        'train/acc',
        'train/acc_cls',
        'train/mean_iu',
        'train/fwavacc',
        'valid/losses',
        'valid/acc',
        'valid/acc_cls',
        'valid/mean_iu',
        'valid/fwavacc',
        'elapsed_time',
    ]

    def __init__(self, out_dir, instance_problem, tensorboard_writer=None):

        # Copies of things the trainer was given access to
        self.instance_problem = instance_problem

        # Helper objects
        self.tensorboard_writer = tensorboard_writer

        # Log directory / log files
        self.out_dir = out_dir
        if not osp.exists(self.out_dir):
            os.makedirs(self.out_dir)

        if not osp.exists(osp.join(self.out_dir, 'log.csv')):
            with open(osp.join(self.out_dir, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        # Logging parameters
        self.timestamp_start = datetime.datetime.now(pytz.timezone(MY_TIMEZONE))

    def write_metrics(self, metrics, loss, split, epoch, iteration):
        with open(osp.join(self.out_dir, 'log.csv'), 'a') as f:
            elapsed_time = (
                    datetime.datetime.now(pytz.timezone(MY_TIMEZONE)) -
                    self.timestamp_start).total_seconds()
            if split == 'val':
                log = [epoch, iteration] + [''] * 5 + \
                      [loss] + list(metrics) + [elapsed_time]
            elif split == 'train':
                try:
                    metrics_as_list = metrics.tolist()
                except:
                    metrics_as_list = list(metrics)
                log = [epoch, iteration] + [loss] + \
                      metrics_as_list + [''] * 5 + [elapsed_time]
            else:
                raise ValueError('split not recognized')
            log = map(str, log)
            f.write(','.join(log) + '\n')

