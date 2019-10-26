import time
import os

from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer
import datetime
import shutil
from instanceseg.train.evaluator import Evaluator
import torch


class CheckpointFileHandler(PatternMatchingEventHandler):
    patterns = ['*.pth.tar', '*.pth']
    queued_prefix = 'queued'
    started_prefix = 'started'
    finished_prefix = 'finished'

    def __init__(self, validator: Evaluator, file_event_logdir):
        self.file_event_logdir = file_event_logdir
        super(CheckpointFileHandler, self).__init__(self)
        self.current_logfile = None
        self.status = None
        self.validator = validator
        self.file_queue = []
        self.logging_file = os.path.join(self.file_event_logdir, 'logging_file')
        with open(self.logging_file, 'w') as f:
            f.write('')  # Write nothing; we'll append to it

    def broadcast(self, msg):
        with open(self.logging_file, 'a') as f:
            f.write(msg + '\n')
        print(msg)

    def broadcast_started(self, checkpoint_file):
        queue_file = os.path.join(self.file_event_logdir,
                                  '{}-watcherlog_{}.txt'.format(self.queued_prefix, os.path.basename(
                                      checkpoint_file)))
        started_file = queue_file.replace(self.queued_prefix, self.started_prefix)
        self.current_logfile = started_file
        msg = "{}\tStarted processing {}".format(datetime.datetime.now(), os.path.basename(checkpoint_file))
        if os.path.exists(queue_file):
            shutil.move(queue_file, started_file)
        with open(self.current_logfile, 'a') as fid:
            fid.write(msg)
        self.broadcast(msg)

    def convert_name_to_finished(self, started_fname):
        return started_fname.replace(self.started_prefix, self.finished_prefix)

    def broadcast_finished(self):
        msg = "\n{}\tFinished processing {}".format(datetime.datetime.now(), os.path.basename(self.current_logfile))
        with open(self.current_logfile, 'a') as fid:
            fid.write('\n')
            fid.write(msg)
        self.broadcast(msg)
        finished_file = self.convert_name_to_finished(self.current_logfile)
        shutil.move(self.current_logfile, finished_file)
        self.current_logfile = finished_file

    def enqueue(self, new_model_pth):
        queue_logfile = os.path.join(self.file_event_logdir,
                                     '{}-watcherlog_{}.txt'.format(self.queued_prefix,
                                                                   os.path.basename(new_model_pth)))
        msg = "{}\tQueued {}".format(datetime.datetime.now(), new_model_pth)
        with open(queue_logfile, 'w') as fid:
            fid.write('\n')
            fid.write(msg)
        self.broadcast(msg)
        self.file_queue.append(new_model_pth)

    def process_new_model_file(self, new_model_pth):
        """
        event.event_type
            'modified' | 'created' | 'moved' | 'deleted'
        event.is_directory
            True | False
        event.src_path
            path/to/observed/file
        """
        self.broadcast_started(new_model_pth)
        checkpoint = torch.load(new_model_pth)
        self.validator.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.validator.state.epoch = checkpoint['epoch']
        self.validator.state.iteration = checkpoint['iteration']
        self.validator.validate_all_splits()
        self.broadcast_finished()

    def on_modified(self, event):
        self.broadcast("{} modified".format(event.src_path))
        if event.src_path in self.file_queue:
            self.broadcast('{} already on queue'.format(event.src_path))
            pass
        else:
            self.enqueue(event.src_path)
        time.sleep(0.5)  # modifications can happen rapidfire -- gets annoying.

    def on_created(self, event):
        pass  # Also creates an on_modified event


class WatchingValidator(object):
    def __init__(self, validator, watch_directory):
        self.watch_directory = watch_directory
        watcher_log_directory = watch_directory.rstrip(os.path.sep) + '-val-log'
        if not os.path.exists(watcher_log_directory):
            os.makedirs(watcher_log_directory)
        self.observer = Observer()
        self.file_handler = CheckpointFileHandler(validator, watcher_log_directory)
        self.observer.schedule(self.file_handler, path=self.watch_directory)
        self.wait_print_sec = 10

    def start(self):
        print('Set up to watch directory {}'.format(self.watch_directory))
        existing_files = os.listdir(self.watch_directory)
        existing_files = sorted([os.path.join(self.watch_directory, f) for f in existing_files
                          if f.endswith('.pth') or f.endswith('.pth.tar')])
        self.file_handler.broadcast('Found files {}'.format([os.path.basename(sf) for sf in existing_files]))
        for file in existing_files:
            self.file_handler.process_new_model_file(file)
        self.observer.start()
        try:
            self.file_handler.broadcast('Starting while() to continuously process files')
            time_since_last_file = 0
            while True:
                if len(self.file_handler.file_queue) > 0:
                    time_since_last_file = 0
                    model_pth_to_process = self.file_handler.file_queue.pop(0)
                    self.file_handler.broadcast('Popped {}'.format(os.path.basename(model_pth_to_process)))
                    self.file_handler.process_new_model_file(model_pth_to_process)
                time.sleep(1)
                time_since_last_file += 1
                if (time_since_last_file % self.wait_print_sec) == 0:
                    print('\r{}m{}s since last file'.format(int(time_since_last_file / 60), time_since_last_file %
                                                            60), end='')
        except KeyboardInterrupt:
            self.observer.stop()
        self.observer.join()
        self.file_handler.broadcast('Exiting observer')
