#!/usr/bin/env python

from __future__ import print_function

from distutils.version import LooseVersion
import sys

from setuptools import find_packages
from setuptools import setup


__version__ = '1.7.2'


if sys.argv[-1] == 'release':
    import shlex
    import subprocess
    commands = [
        'python setup.py sdist',
        'twine upload dist/torchfcn-{0}.tar.gz'.format(__version__),
        'git tag v{0}'.format(__version__),
        'git push origin master --tags',
    ]
    for cmd in commands:
        subprocess.call(shlex.split(cmd))
    sys.exit(0)


try:
    import torch  # NOQA
    if LooseVersion(torch.__version__) < LooseVersion('0.2.0'):
        raise ImportError
except ImportError:
    print('Please install pytorch>=0.2.0. (See http://pytorch.org)',
          file=sys.stderr)
    sys.exit(1)


setup(
    name='torchfcn',
    version=__version__,
    packages=find_packages(),
    install_requires=[r.strip() for r in open('requirements.txt')],
    description='pytorch implementation of fully convolutional networks.',
    package_data={'torchfcn': ['ext/*']},
    include_package_data=true,
    author='kentaro wada',
    author_email='www.kentaro.wada@gmail.com',
    license='mit',
    url='https://github.com/wkentaro/pytorch-fcn',
    classifiers=[
        'development status :: 5 - production/stable',
        'intended audience :: developers',
        'license :: osi approved :: mit license',
        'operating system :: posix',
        'topic :: internet :: www/http',
    ],
)
