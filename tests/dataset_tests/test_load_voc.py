import instanceseg.factory.data
from scripts.configurations import voc_cfg


if __name__ == '__main__':
    cfg = voc_cfg.get_default_config()
    cfg['ordering'] = 'LR'
    print('Getting datasets')
    train_dataset, val_dataset = instanceseg.factory.data.get_datasets_with_transformations('voc', cfg)
    num_images = len(train_dataset)
    print('Loaded {}/{}'.format(0, num_images))
    for idx, (img, (sem_lbl, inst_lbl)) in enumerate(train_dataset):
        if divmod(idx+1, 100)[1] == 0:
            print('Loaded {}/{}'.format(idx+1, num_images))
    print('Loaded {}/{}'.format(num_images, num_images))
