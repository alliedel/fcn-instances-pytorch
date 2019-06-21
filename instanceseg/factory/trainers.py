from tensorboardX import SummaryWriter

from instanceseg.train import trainer


def get_trainer(cfg, cuda, model, optim, dataloaders, problem_config, out_dir, scheduler=None):
    writer = SummaryWriter(log_dir=out_dir)
    trainer = trainer.Trainer(cuda=cuda, model=model, optimizer=optim, train_loader=dataloaders['train'],
                              val_loader=dataloaders['val'], out_dir=out_dir, max_iter=cfg['max_iteration'],
                              instance_problem=problem_config, size_average=cfg['size_average'],
                              interval_validate=cfg.get('interval_validate', len(dataloaders['train'])),
                              loss_type=cfg['loss_type'], matching_loss=cfg['matching'], tensorboard_writer=writer,
                              train_loader_for_val=dataloaders['train_for_val'],
                              augment_input_with_semantic_masks=cfg['augment_semantic'],
                              export_activations=cfg['export_activations'],
                              activation_layers_to_export=cfg['activation_layers_to_export'],
                              write_instance_metrics=cfg['write_instance_metrics'],
                              generate_new_synthetic_data_each_epoch=(
                                      cfg['dataset'] == 'synthetic' and cfg['infinite_synthetic']),
                              lr_scheduler=scheduler)
    return trainer
