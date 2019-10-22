import instanceseg.utils.parse
from scripts import train
from instanceseg.utils import script_setup as script_utils


if __name__ == '__main__':
    config = dict(dataset_name='synthetic',
                  gpu=1,
                  config_idx=0,
                  sampler_name=None)

    commandline_arguments_list = utils.parse.construct_args_list_to_replace_sys(**config)

    train.main(commandline_arguments_list)
