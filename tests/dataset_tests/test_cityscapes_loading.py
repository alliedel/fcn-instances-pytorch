from scripts.configurations import cityscapes_cfg
import torchfcn.utils.data


if __name__ == '__main__':
    cfg = cityscapes_cfg.get_default_config()
    cfg['ordering'] = None  # 'LR'
    print('Getting datasets')
    train_dataset, val_dataset = torchfcn.utils.data.get_datasets_with_transformations('cityscapes', cfg)
    num_images = len(train_dataset)
    print('Loaded {}/{}'.format(0, num_images))
    import ipdb; ipdb.set_trace()
    train_dataset.__getitem__(0)
    for idx, (img, (sem_lbl, inst_lbl)) in enumerate(train_dataset):
        if divmod(idx+1, 100)[1] == 0:
            print('Loaded {}/{}'.format(idx+1, num_images))
    print('Loaded {}/{}'.format(num_images, num_images))
