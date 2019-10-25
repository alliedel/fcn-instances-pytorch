from instanceseg.utils import parse
from scripts import train
from instanceseg.utils import script_setup as script_utils


if __name__ == '__main__':
    # config = dict(dataset_name='synthetic',
    #               gpu=[0, 1],
    #               config_idx=0,
    #               sampler_name=None)
    config = dict(dataset_name='cityscapes',
                  gpu=[0, 1],
                  config_idx='overfit_above_capacity',
                  sampler_name=None,
                  max_iteration=100)

    commandline_arguments_list = parse.construct_args_list_to_replace_sys(**config)

    train.main(commandline_arguments_list)
