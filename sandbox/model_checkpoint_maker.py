import numpy as np
import os
import shutil


class ModelHistorySaver(object):
    def __init__(self, model_checkpoint_dir, interval_validate, max_n_saved_models=20, max_n_iterations=100000):
        assert np.mod(max_n_saved_models, 2) == 0, 'Max_n_saved_models must be even'
        self.model_checkpoint_dir = model_checkpoint_dir
        if not os.path.exists(model_checkpoint_dir):
            raise Exception('{} does not exist'.format(model_checkpoint_dir))
        self.interval_validate = interval_validate
        self.max_n_saved_models = max_n_saved_models
        n_digits = max(6, np.ceil(np.log10(max_n_iterations + 1)))
        self.itr_format = '{:0' + str(n_digits) + 'd}'
        self.adaptive_save_model_every = 1

    def get_list_of_checkpoint_files(self):
        return [os.path.join(self.model_checkpoint_dir, f) for f in sorted(os.listdir(self.model_checkpoint_dir))]

    def get_latest_checkpoint_file(self):
        return self.get_list_of_checkpoint_files()[-1]

    def get_model_filename_from_iteration(self, i):
        return os.path.join(self.model_checkpoint_dir, 'model_' + self.itr_format.format(i) + '.txt')

    def get_iteration_from_model_filename(self, model_filename):
        itr_as_06d = os.path.basename(model_filename).split('_')[1].split('.')[0]
        assert itr_as_06d.isdigit()
        return int(itr_as_06d)

    def save_model_to_history(self, current_itr):
        if np.mod(current_itr, self.adaptive_save_model_every * self.interval_validate) == 0:
            filename = self.get_model_filename_from_iteration(current_itr)
            fid = open(filename, 'w')
            fid.write('State dict here')
            self.clean_up_checkpoints()
            return True
        else:
            return False

    def clean_up_checkpoints(self):
        """
        Cleans out history to keep only a small number of models; always ensures we keep the first and most recent.
        """
        most_recent_file = self.get_latest_checkpoint_file()
        most_recent_itr = self.get_iteration_from_model_filename(most_recent_file)
        n_vals_so_far = most_recent_itr / self.interval_validate
        if (n_vals_so_far / self.adaptive_save_model_every) >= (self.max_n_saved_models):
            print('Cleaning up at iteration {}, with {} files'.format(most_recent_itr,
                                                                      len(self.get_list_of_checkpoint_files())))
            while (n_vals_so_far / self.adaptive_save_model_every) >= self.max_n_saved_models:
                self.adaptive_save_model_every *= 2  # should use ceil, log2 to compute instead (this is hacky)
            iterations_to_keep = range(0, most_recent_itr + self.interval_validate,
                                       self.adaptive_save_model_every * self.interval_validate)
            if most_recent_itr not in iterations_to_keep:
                iterations_to_keep.append(most_recent_itr)
            for j in iterations_to_keep:  # make sure the files we assume exist actually exist
                assert os.path.exists(self.get_model_filename_from_iteration(j)), \
                    '{} does not exist'.format(f)

            for model_file in self.get_list_of_checkpoint_files():
                iteration_number = self.get_iteration_from_model_filename(model_file)
                if iteration_number not in iterations_to_keep:
                    os.remove(model_file)
            assert len(self.get_list_of_checkpoint_files()) <= (self.max_n_saved_models + 1), 'DebugError'


def main():
    model_checkpoint_dir = 'sandbox/model_checkpoints'
    max_n_iterations = 100000
    if os.path.exists(model_checkpoint_dir):
        shutil.rmtree(model_checkpoint_dir)
    os.mkdir(model_checkpoint_dir)
    model_history_saver = ModelHistorySaver(model_checkpoint_dir=model_checkpoint_dir,
                                            interval_validate=100,
                                            max_n_saved_models=20,
                                            max_n_iterations=max_n_iterations)
    for current_itr in range(int(max_n_iterations / 10)):
        if np.mod(current_itr, model_history_saver.interval_validate) == 0:
            model_history_saver.save_model_to_history(current_itr)


if __name__ == '__main__':
    main()
