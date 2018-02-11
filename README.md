# Instance Segmentation, built on top of pytorch-fcn

## Installation
Not third-party friendly at the moment (sorry!).  But still should be possible --

You'll have to clone this and add it to your python path: https://bitbucket.org/alliedel/my_tools
(I believe only local_pyutils.py is needed for this project).

All packages I have are listed in the requirements.txt file; not all are actually required, so you could alternatively just install packages and rerun until it's happy :)

If you're trying to run on PASCAL, you're likely to run into some hard-coded paths.  Hopefully simply pointing it to the right place resolves that (but I'm not super confident)

## Scripts
Each of the scripts requires an argument for the GPU number to run on.  If you'd like to run on CPU, set -g to -1.
### Run a synthetic example:
`python examples/pink_blobs/train_fcn8s_pink_blobs.py -g 0`

### Run on PASCAL VOC:
`python examples/voc/train_fcn8s_all_voc_instances.py -g 0`

## Pointers to the interesting parts of the code
Matching loss:
torchfcn/losses.py

Network architecture:
torchfcn/models/fcn8s_instance.py
