from tensorboardX import SummaryWriter

from instanceseg.train.trainer import Trainer


def get_trainer(cfg, cuda, model, dataloaders, problem_config, out_dir, optim, scheduler=None):
    writer = SummaryWriter(log_dir=out_dir)
    trainer = Trainer(cuda=cuda, model=model, optimizer=optim, dataloaders=dataloaders,
                      out_dir=out_dir, max_iter=cfg['max_iteration'],
                      instance_problem=problem_config, size_average=cfg['size_average'],
                      interval_validate=cfg.get('interval_validate', len(dataloaders['train'])),
                      loss_type=cfg['loss_type'], matching_loss=cfg['matching'], tensorboard_writer=writer,
                      augment_input_with_semantic_masks=cfg['augment_semantic'],
                      export_activations=cfg['export_activations'],
                      activation_layers_to_export=cfg['activation_layers_to_export'],
                      write_instance_metrics=cfg['write_instance_metrics'],
                      generate_new_synthetic_data_each_epoch=(
                              cfg['dataset'] == 'synthetic' and cfg['infinite_synthetic']),
                      lr_scheduler=scheduler)
    return trainer


def get_evaluator(cfg, cuda, model, dataloaders, problem_config, out_dir):
    writer = SummaryWriter(log_dir=out_dir)
    trainer = Trainer(
        size_average=None, max_iter=None, optimizer=None, loss_type=None, interval_validate=None, lr_scheduler=None,
        dataloaders=dataloaders,
        cuda=cuda, model=model, out_dir=out_dir,
        instance_problem=problem_config,
        tensorboard_writer=writer,
        augment_input_with_semantic_masks=cfg['augment_semantic'],
        export_activations=cfg['export_activations'],
        activation_layers_to_export=cfg['activation_layers_to_export'],
        write_instance_metrics=cfg['write_instance_metrics'],
        generate_new_synthetic_data_each_epoch=(
                cfg['dataset'] == 'synthetic' and cfg['infinite_synthetic'])
        )
    return trainer
