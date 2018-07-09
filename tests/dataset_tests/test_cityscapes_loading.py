from scripts.configurations import cityscapes_cfg
import torchfcn.utils.data
import tqdm


if __name__ == '__main__':
    cfg = cityscapes_cfg.get_default_config()
    cfg['ordering'] = None  # 'LR'
    print('Getting datasets')
    train_dataset, val_dataset = torchfcn.utils.data.get_datasets_with_transformations('cityscapes', cfg)
    num_images = len(train_dataset)
    print('Loaded {}/{}'.format(0, num_images))
    import ipdb; ipdb.set_trace()
    train_dataset.__getitem__(0)
    for idx, (img, (sem_lbl, inst_lbl)) in tqdm.tqdm(enumerate(train_dataset), desc='Loading Cityscape images',
                                                     total=len(train_dataset)):
        pass
