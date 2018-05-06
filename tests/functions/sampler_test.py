import os
import torch
import tqdm
from torchfcn.datasets import dataset_utils
from torchfcn.datasets import voc
from torchfcn import script_utils
from scripts.configurations import voc_cfg


def main():
    # Setup
    cfg = voc_cfg.default_config
    train_dataset, val_dataset = script_utils.get_voc_datasets(cfg, '/home/adelgior/data/datasets/')
    gpu = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    # Get sampler
    train_sampler = dataset_utils.sampler_factory(sequential=True)(train_dataset)

    # Apply sampler to dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, sampler=train_sampler,
                                               **loader_kwargs)

    for idx, (d, (sem_lbl, inst_lbl)) in tqdm.tqdm(enumerate(train_loader), desc='Iterating through new dataloader',
                                                   total=len(train_loader)):
        pass


if __name__ == '__main__':
    main()
