import argparse
import os.path

from instanceseg.utils import configs
from scripts.configurations.sampler_cfg_registry import sampler_cfgs
from instanceseg.datasets import dataset_registry


def parse_args_test(replacement_args_list=None):
    parser = get_parser_test()
    args, argv = parser.parse_known_args(replacement_args_list) \
        if replacement_args_list is not None else parser.parse_known_args()

    # Config override parser
    assert args.dataset is not None, ValueError('dataset argument must not be None.  '
                                                'Run with --help for more details.')
    dataset_name_from_logdir = os.path.split(os.path.split(args.logdir)[0])[1]
    assert args.dataset == dataset_name_from_logdir, 'Dataset given: {}.  I expected this dataset from logdir: ' \
                                                     '{}'.format(args.dataset,
                                                                 os.path.split(os.path.split(args.logdir)[0])[1])

    cfg_default = dataset_registry.REGISTRY[args.dataset].default_config
    cfg_override_parser = configs.get_cfg_override_parser(cfg_default)

    bad_args = [arg for arg in argv[::2] if arg.replace('-', '') not in cfg_default.keys()]
    assert len(bad_args) == 0, cfg_override_parser.error('bad_args: {}'.format(bad_args))
    if args.sampler is not None:
        argv += ['--sampler', args.sampler]
    # Parse with list of options
    override_cfg_args, leftovers = cfg_override_parser.parse_known_args(argv)
    assert len(leftovers) == 0, ValueError('args not recognized: {}'.format(leftovers))
    # apparently this is failing, so I'm going to have to screen this on my own:

    # Remove options from namespace that weren't defined
    unused_keys = [k for k in list(override_cfg_args.__dict__.keys()) if
                   '--' + k not in argv and '-' + k not in argv]
    for k in unused_keys:
        delattr(override_cfg_args, k)

    postprocess_test_args(override_cfg_args)
    return args, override_cfg_args


def parse_args_train(replacement_args_list=None):
    # Get initial parser
    parser = get_parser_train()
    args, argv = parser.parse_known_args(replacement_args_list) \
        if replacement_args_list is not None else parser.parse_known_args()

    # Config override parser
    assert args.dataset is not None, ValueError('dataset argument must not be None.  '
                                                'Run with --help for more details.')
    cfg_default = dataset_registry.REGISTRY[args.dataset].default_config
    cfg_override_parser = configs.get_cfg_override_parser(cfg_default)

    bad_args = [arg for arg in argv[::2] if arg.replace('-', '') not in cfg_default.keys()]
    assert len(bad_args) == 0, cfg_override_parser.error('bad_args: {}'.format(bad_args))
    if args.sampler is not None:
        argv += ['--sampler', args.sampler]
    # Parse with list of options
    override_cfg_args, leftovers = cfg_override_parser.parse_known_args(argv)
    assert len(leftovers) == 0, ValueError('args not recognized: {}'.format(leftovers))
    # apparently this is failing, so I'm going to have to screen this on my own:

    # Remove options from namespace that weren't defined
    unused_keys = [k for k in list(override_cfg_args.__dict__.keys()) if
                   '--' + k not in argv and '-' + k not in argv]
    for k in unused_keys:
        delattr(override_cfg_args, k)

    postprocess_train_args(override_cfg_args)

    return args, override_cfg_args


def postprocess_test_args(override_cfg_args):

    pass


def postprocess_train_args(override_cfg_args):
    # Fix a few values
    replace_attr_with_function_of_val(override_cfg_args, 'clip',
                                      lambda old_val: old_val if old_val > 0 else None,
                                      error_if_attr_doesnt_exist=False)
    replace_attr_with_function_of_val(override_cfg_args, 'semantic_subset',
                                      lambda old_val: convert_comma_separated_string_to_list(
                                          old_val, str),
                                      error_if_attr_doesnt_exist=False)
    replace_attr_with_function_of_val(override_cfg_args, 'img_size',
                                      lambda old_val: convert_comma_separated_string_to_list(
                                          old_val, int),
                                      error_if_attr_doesnt_exist=False)
    replace_attr_with_function_of_val(override_cfg_args, 'resize_size',
                                      lambda old_val: convert_comma_separated_string_to_list(
                                          old_val, int),
                                      error_if_attr_doesnt_exist=False)


def construct_args_list_to_replace_sys(dataset_name, gpu=None, config_idx=None, sampler_name=None,
                                       resume=None, **kwargs):
    default_args_list = [dataset_name]
    for key, val in {'-g': gpu, '-c': config_idx, '--sampler': sampler_name,
                     '--resume': resume}.items():
        if val is not None:
            default_args_list += [key, str(val)]
    for key, val in kwargs:
        if val is not None:
            default_args_list += [key, str(val)]
    return default_args_list


def get_parser_train():
    parser = argparse.ArgumentParser()
    dataset_names = dataset_registry.REGISTRY.keys()
    subparsers = parser.add_subparsers(help='dataset: {}'.format(dataset_names), dest='dataset')
    dataset_parsers = {
        dataset_name:
            subparsers.add_parser(dataset_name, help='{} dataset options'.format(dataset_name),
                                  epilog='\n\nOverride options:\n' + '\n'.join(
                                      ['--{}: {}'.format(k, v)
                                       for k, v in dataset_registry.REGISTRY[
                                           dataset_name].default_config.items()]),
                                  formatter_class=argparse.RawTextHelpFormatter)
        for dataset_name in dataset_names
    }
    for dataset_name, subparser in dataset_parsers.items():
        cfg_choices = list(dataset_registry.REGISTRY[dataset_name].config_options.keys())
        subparser.add_argument('-c', '--config', type=str_or_int, default=0, choices=cfg_choices)
        subparser.add_argument('-g', '--gpu', type=int, nargs='+', required=True)
        subparser.add_argument('--resume', help='Checkpoint path')
        subparser.add_argument('--semantic-init',
                               help='Checkpoint path of semantic model (e.g. - '
                                    '\'~/data/models/pytorch/semantic_synthetic.pth\'',
                               default=None)
        subparser.add_argument('--single-image-index', type=int,
                               help='Image index to use for train/validation set',
                               default=None)
        subparser.add_argument('--sampler', type=str, choices=sampler_cfgs.keys(), default=None,
                               help='Sampler for dataset')
    return parser


def get_parser_test():
    parser = argparse.ArgumentParser()
    dataset_names = dataset_registry.REGISTRY.keys()
    subparsers = parser.add_subparsers(help='dataset: {}'.format(dataset_names), dest='dataset')
    dataset_parsers = {
        dataset_name:
            subparsers.add_parser(dataset_name, help='{} dataset options'.format(dataset_name),
                                  epilog='\n\nOverride options:\n' + '\n'.join(
                                      ['--{}: {}'.format(k, v)
                                       for k, v in dataset_registry.REGISTRY[
                                           dataset_name].default_config.items()]),
                                  formatter_class=argparse.RawTextHelpFormatter)
        for dataset_name in dataset_names
    }
    for dataset_name, subparser in dataset_parsers.items():
        cfg_choices = list(dataset_registry.REGISTRY[dataset_name].config_options.keys())
        subparser.add_argument('-c', '--config', type=str_or_int, default=0, choices=cfg_choices)
        subparser.add_argument('-g', '--gpu', help='ex. - \'2 3\'', type=int, nargs='+', required=True)
        subparser.add_argument('--logdir', help='Checkpoint path for model', required=True)
        subparser.add_argument('--single-image-index', type=int,
                               help='Image index to use for unit testing',
                               default=None)
        subparser.add_argument('--sampler', choices=sampler_cfgs.keys(), default=None,
                               help='Sampler for dataset')
        subparser.add_argument('--test_split', type=str, default='val')
    return parser


def parse_args_without_sys(dataset_name, gpu=0, **kwargs):
    replacement_args_list = construct_args_list_to_replace_sys(dataset_name, gpu=gpu, **kwargs)
    print(replacement_args_list)
    args, override_cfg_args = parse_args_train(replacement_args_list=replacement_args_list)
    return args, override_cfg_args


def str_or_int(val):
    try:
        return int(val)
    except ValueError:
        return val


def replace_attr_with_function_of_val(namespace, attr, replacement_function,
                                      error_if_attr_doesnt_exist=True):
    if attr in namespace.__dict__.keys():
        setattr(namespace, attr, replacement_function(getattr(namespace, attr)))
    elif error_if_attr_doesnt_exist:
        raise Exception('attr {} does not exist in namespace'.format(attr))


def convert_comma_separated_string_to_list(string, conversion_type=None):
    if conversion_type is None:
        conversion_type = str
    if string is None or string == '':
        return string
    else:
        elements = [s.strip() for s in string.split(',')]
        return [conversion_type(element) for element in elements]
