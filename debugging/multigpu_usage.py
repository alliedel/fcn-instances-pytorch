from instanceseg.utils import parse
from scripts import train


if __name__ == '__main__':
    config = dict(dataset_name='cityscapes',
                  gpu='0 2',
                  config_idx=0,
                  sampler_name=None)

    commandline_arguments_list = parse.construct_args_list_to_replace_sys(**config)
    import ipdb; ipdb.set_trace()
    train.main(commandline_arguments_list)

