from instanceseg.utils import script_setup
import time
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler


class CheckpointFileHandler(PatternMatchingEventHandler):
    patterns = ["*.pth.tar", "*.pth"]

    def process_file(self, event):
        """
        event.event_type
            'modified' | 'created' | 'moved' | 'deleted'
        event.is_directory
            True | False
        event.src_path
            path/to/observed/file
        """

        print('{} processed!'.format(event.src_path))

    def on_modified(self, event):
        print('{} modified'.format(event.src_path))
        self.process_file(event)

    def on_created(self, event):
        pass  # Also creates an on_modified event


class WatchingValidator(object):
    def __init__(self, evaluator, watch_directory=None):
        self.evaluator = evaluator
        if watch_directory is None:
            self.watch_directory = self.evaluator.exporter.model_history_saver.model_checkpoint_dir
        else:
            self.watch_directory = watch_directory
        self.observer = Observer()
        self.file_handler = CheckpointFileHandler()
        self.observer.schedule(self.file_handler, path=self.watch_directory)

    def start(self):
        self.observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()

        self.observer.join()


def create_val_watcher(dataset, cfg, out_dir, sampler_cfg, init_model_checkpoint_path, gpu):
    script_setup.setup_val_watcher(dataset_type=dataset, cfg=cfg, out_dir=out_dir,
                                   sampler_cfg=sampler_cfg,
                                   init_model_checkpoint_path=init_model_checkpoint_path, gpu=gpu)


if __name__ == '__main__':
    watching_validator = WatchingValidator(evaluator=None, watch_directory='/tmp/watch_directory')
    watching_validator.start()
