from scripts import train_instances_filtered
from instanceseg.utils import scripts as script_utils


if __name__ == '__main__':
    config = dict(dataset_name='synthetic',
                  gpu=1,
                  config_idx=0,
                  sampler_name=None)

    commandline_arguments_list = script_utils.construct_args_list_to_replace_sys(**config)

    train_instances_filtered.main(commandline_arguments_list)
