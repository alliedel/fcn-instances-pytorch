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

    def __init__(self, validator: Evaluator, file_event_logdir):
        self.file_event_logdir = file_event_logdir
        super(CheckpointFileHandler, self).__init__(self)
        self.current_logfile = None
        self.status = None
        self.validator = validator

    def broadcast_started(self, checkpoint_file):
        self.current_logfile = os.path.join(self.file_event_logdir,
                                            'started-watcherlog_{}.txt'.format(os.path.basename(checkpoint_file)))
        msg = "{}\tStarted processing {}".format(datetime.datetime.now(), checkpoint_file)
        with open(self.current_logfile, 'w+') as fid:
            fid.write(msg)
        print(msg)

    def convert_name_to_finished(self, started_fname):
        return started_fname.replace('started-', 'finished-')

    def broadcast_finished(self):
        msg = "{}\tFinished processing {}".format(datetime.datetime.now(), self.current_logfile)
        with open(self.current_logfile, 'w+') as fid:
            fid.write(msg)
        finished_file = self.convert_name_to_finished(self.current_logfile)
        shutil.copyfile(self.current_logfile, finished_file)
        self.current_logfile = finished_file

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
        self.validator.validate_all_splits()
        self.broadcast_finished()

    def on_modified(self, event):
        print("{} modified".format(event.src_path))
        self.process_new_model_file(event.src_path)

    def on_created(self, event):
        pass  # Also creates an on_modified event


class WatchingValidator(object):
    def __init__(self, validator, watch_directory):
        self.watch_directory = watch_directory
        self.observer = Observer()
        watcher_log_directory = watch_directory.rstrip(os.path.sep) + '-val-log'
        if not os.path.exists(watcher_log_directory):
            os.makedirs(watcher_log_directory)
        self.file_handler = CheckpointFileHandler(validator, watcher_log_directory)
        self.observer.schedule(self.file_handler, path=self.watch_directory)

    def start(self):
        print('Set up to watch directory {}'.format(self.watch_directory))
        existing_files = os.listdir(self.watch_directory)
        existing_files = [os.path.join(self.watch_directory, f) for f in existing_files
                          if f.endswith('.pth') or f.endswith('.pth.tar')]
        print('Found files {}'.format(existing_files))
        for file in existing_files:
            self.file_handler.process_new_model_file(file)
        self.observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()

        self.observer.join()
