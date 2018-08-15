import os
import fnmatch, re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'

file_comments = {
    ''
}


def match_regex(regex, string):
    return bool(re.compile(fnmatch.translate(regex)).match(string))


def bool_print_filename(filename_from_root=None):
    basename = os.path.basename(filename_from_root)
    extensions_to_ignore = ['.pyc', '.jpg', '.prototxt', '.txt', '.png']
    extensions_to_print = ['.py']
    basenames_to_ignore = ['__init__.py']
    basename_regexs_to_ignore = ['fcn*s.py']
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
    return True


def bool_print_directory(directory_from_root):
    """
    We assume that any directory you do NOT want to print, you also do not want to enter.
    So if this returns false, the directory will not be printed AND will be removed from os.walk.
    If this return true, we will generally enter the directory, unless bool_print_but_dont_enter_directory returns True.
    """
    subdir = os.path.basename(directory_from_root)
    subdirectories_to_ignore = ['__pycache__', '.idea', 'cmake-build-debug']
    regexs_to_ignore = ['*logs*']
    fullpaths_to_ignore = []
    if any([match_regex(r, subdir) for r in regexs_to_ignore]):
        return False
    if any([subdir.rstrip(os.sep) == s.rstrip(os.sep) for s in subdirectories_to_ignore]):
        return False
    if any([directory_from_root == p.rstrip(os.sep) for p in fullpaths_to_ignore]):
        return False
    # print directory by default
    return True


def bool_print_but_dont_enter_directory(directory_from_root):
    subdirectories_to_ignore = []
    regexs_to_ignore = ['instanceseg/ext']

    if any([match_regex(r, directory_from_root) for r in regexs_to_ignore]):
        return False
    if any([subdir.rstrip(os.sep) == s.rstrip(os.sep) for s in subdirectories_to_ignore]):
        return False
    if any([directory_from_root == p.rstrip(os.sep) for p in fullpaths_to_ignore]):
        return False
        # print directory by default
    return False


def bool_enter_directory(dir_from_root):
    if not bool_print_directory(dir_from_root):  # Dont print implies don't enter
        return False
    elif bool_print_but_dont_enter_directory(dir_from_root):  # If print, maybe still don't enter after
        return False
    else:
        return True


def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        dirs.sort()
        files.sort()
        path_from_startdir = root.replace(startpath, '')
        level = path_from_startdir.count(os.sep)
        indent = ''.join(['|    '] * (level)) + '|--'

        for d in dirs:
            full_d = os.path.join(path_from_startdir, d)
            if bool_print_but_dont_enter_directory(full_d):
                print(os.path.join('{}{}', '').format(indent, d))

        dirs[:] = [d for d in dirs if bool_enter_directory(os.path.join(path_from_startdir, d))]  # in-place

        if bool_print_directory(path_from_startdir):  # This check is only really necessary for the first level (
            # after that, we just wont enter these directories)
            print(os.path.join('{}{}', '').format(indent, os.path.basename(path_from_startdir)))

        subindent = ''.join(['|    '] * (level + 1)) + '|-- '
        for f in files:
            if bool_print_filename(os.path.join(path_from_startdir, f)):
                print('{}{}'.format(subindent, f))


def main():
    print(__file__)
    print('Listing files in project root {}'.format(PROJECT_ROOT))
    list_files(PROJECT_ROOT)


if __name__ == '__main__':
    main()
