import argparse
import fnmatch
import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'

file_comments = {
    'graveyard': 'old reference code',
    'instanceseg': 'MAIN MODULE - core of the implementation',
    'instanceseg/datasets': 'interfaces for datasets, as well as common transformations '
                            '(runtime and precomputed) and image samplers',
    'instanceseg/datasets/dataset_registry.py': 'new datasets need to be registered here',
    'instanceseg/datasets/dataset_statistics.py': 'Computes number/size of instances, etc.  '
                                                  'Used for dataset analysis as well as samplers (filtering images '
                                                  'by image stats)',
    'instanceseg/ext': 'external (third party) code',
    'instanceseg/xentropy.py': 'losses functions (cross entropy, etc.)',
    'instanceseg/metrics.py': 'various intermediate metrics for analysis (e.g. - channel usage)',
    'instanceseg/trainer.py': 'implements the Trainer, which orchestrates model training/validation',
    'instanceseg/factory': 'generates configured instances of various objects '
                           '(datasets, samplers, models, trainers)',
    'instanceseg/models': 'architectures for instance segmentation',
    'instanceseg/utils': 'lightweight functions used by various classes',
    'tests': 'unit tests and pipeline tests - also good to reference for sample code',
    'scripts/train_instances_filtered.py': 'MAIN SCRIPT - trains an instance segmentation model',
    'scripts/analysis': 'dataset/log analysis (post-experiment)',
    'scripts/configurations': 'experiment configurations -- organized by dataset.',
}


def get_file_comments():
    return file_comments.copy()


def get_comment_if_exists(file_or_path_from_root, comments_dict):
    if file_or_path_from_root in comments_dict.keys():
        return '# ' + comments_dict.pop(file_or_path_from_root)
    else:
        return ''


def add_comment(string, comment, comment_start_column=None):
    if comment_start_column is None:
        comment_start_column = len(string) + 2
    if len(string) > comment_start_column:  # cut off string if necessary
        string = string[:comment_start_column]

    string = string + ' ' * (comment_start_column - len(string))  # add indent
    return string + comment


def match_regex(regex, string):
    return bool(re.compile(fnmatch.translate(regex)).match(string))


def bool_print_filename(filename_from_root=None, default=True):
    basename = os.path.basename(filename_from_root)
    extensions_to_ignore = ['.pyc', '.jpg', '.prototxt', '.txt', '.png']
    extensions_to_print = ['.py']
    basenames_to_ignore = ['__init__.py']
    basename_regexs_to_ignore = ['fcn*s.py', '.gitignore']
    fullfile_regexs_to_ignore = []
    if any([match_regex(r, filename_from_root) for r in basename_regexs_to_ignore]):
        return False
    elif any([match_regex(r, basename) for r in fullfile_regexs_to_ignore]):
        return False
    elif any([basename.endswith(ext) for ext in extensions_to_ignore]):
        return False
    elif any([basename == b for b in basenames_to_ignore]):
        return False
    elif any([basename.endswith(ext) for ext in extensions_to_print]):
        return True
    # print file by default
    return default


def bool_print_directory(directory_from_root, default=True):
    """
    We assume that any directory you do NOT want to print, you also do not want to enter.
    So if this returns false, the directory will not be printed AND will be removed from os.walk.
    If this return true, we will generally enter the directory, unless bool_print_but_dont_enter_directory returns True.
    """
    subdir = os.path.basename(directory_from_root)
    subdirectories_to_ignore = ['__pycache__', '.idea', 'cmake-build-debug', '.git']
    regexs_to_ignore = ['*logs*']
    fullpaths_to_ignore = []
    if any([match_regex(r, subdir) for r in regexs_to_ignore]):
        return False
    if any([subdir.rstrip(os.sep) == s.rstrip(os.sep) for s in subdirectories_to_ignore]):
        return False
    if any([directory_from_root == p.rstrip(os.sep) for p in fullpaths_to_ignore]):
        return False
    # print directory by default
    return default


def bool_print_but_dont_enter_directory(directory_from_root, default=False):
    subdirectories_to_ignore = []
    regexs_to_ignore = ['instanceseg/ext', 'graveyard', 'tests', 'scripts/analysis', 'scripts/configurations']
    subdir = os.path.basename(directory_from_root)
    if any([match_regex(r, directory_from_root) for r in regexs_to_ignore]):
        return True
    if any([subdir.rstrip(os.sep) == s.rstrip(os.sep) for s in subdirectories_to_ignore]):
        return True
        # print directory by default
    return default


def bool_enter_directory(dir_from_root, default=True):
    if not bool_print_directory(dir_from_root):  # Dont print implies don't enter
        return False
    elif bool_print_but_dont_enter_directory(dir_from_root):  # If print, maybe still don't enter after
        return False
    return default


def print_and_add_to_buffer(string, buffer, print_to_screen=True):
    if print_to_screen:
        print(string)
    buffer += [string]


def get_comment_start_column(startpath, conservative=False):
    buffer = list_files(startpath, conservative=False, comment_start_column=None, print_to_screen=False,
                        include_comments=False)
    return max([len(s) for s in buffer]) + 2


def list_files(startpath, conservative=False, comment_start_column=None, print_to_screen=True,
               include_comments=True):
    buffer = []
    if include_comments:
        comments_dict = get_file_comments()
    if include_comments and comment_start_column is None:
        comment_start_column = get_comment_start_column(startpath, conservative=conservative)
    for root, dirs, files in os.walk(startpath):
        dirs.sort()
        files.sort()
        path_from_startdir = root.replace(startpath, '')

        level = path_from_startdir.count(os.sep)
        indent = ''.join(['|    '] * (level)) + '|--'

        dirs[:] = [d for d in dirs if bool_print_directory(os.path.join(path_from_startdir, d))]  # in-place

        if path_from_startdir == '':  # root -- don't print.  Don't indent filenames.
            subindent = '|-- '
            for f in files:
                if bool_print_filename(os.path.join(path_from_startdir, f)):
                    filename = '{}{}'.format(subindent, f)
                    filename_with_comment = filename if not include_comments \
                        else add_comment(filename, get_comment_if_exists(path_from_startdir, comments_dict),
                                         comment_start_column)
                    print_and_add_to_buffer(filename_with_comment, buffer, print_to_screen)
            continue

        if bool_print_directory(path_from_startdir) or \
                bool_print_but_dont_enter_directory(path_from_startdir):
            filename = os.path.join('{}{}', '').format(indent, os.path.basename(path_from_startdir))
            print_and_add_to_buffer(filename if not include_comments else add_comment(filename, get_comment_if_exists(
                path_from_startdir, comments_dict), comment_start_column), buffer, print_to_screen)

        if not bool_enter_directory(path_from_startdir):
            dirs[:] = []
            continue

        subindent = ''.join(['|    '] * (level + 1)) + '|-- '
        for f in files:
            if bool_print_filename(os.path.join(path_from_startdir, f)):
                filename = '{}{}'.format(subindent, f)
                print_and_add_to_buffer(filename if not include_comments
                                        else
                                        add_comment(filename, get_comment_if_exists(os.path.join(path_from_startdir, f),
                                                                                    comments_dict),
                                                    comment_start_column), buffer, print_to_screen)

    if include_comments:
        leftover_comments = comments_dict.keys()
        if leftover_comments:
            print('Warning: {} comments were leftover:'.format(len(leftover_comments)))
            print(comments_dict)
    return buffer


def main(conservative=False):
    print(__file__)
    print('Listing files in project root {}'.format(PROJECT_ROOT))
    list_files(PROJECT_ROOT, conservative=conservative)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conservative', default=False, action='store_true')
    args = parser.parse_args()
    main(args.conservative)
