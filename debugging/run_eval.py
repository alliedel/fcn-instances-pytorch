from instanceseg.utils import parse
from scripts import evaluate
from instanceseg.utils import script_setup as script_utils


if __name__ == '__main__':
    checkpoint_path = '/usr0/home/adelgior/code/experimental-instanceseg/scripts/logs/synthetic/' \
                      'train_instances_filtered_2019-06-24-163353_VCS-8df0680'
    config = dict(dataset_name='synthetic',
                  resume=checkpoint_path,
                  gpu=1,
                  config_idx=0,
                  sampler_name=None)

    commandline_arguments_list = parse.construct_args_list_to_replace_sys(**config)

    evaluate.main(commandline_arguments_list)
