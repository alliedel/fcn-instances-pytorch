from tensorboardX import SummaryWriter

import instanceseg
from instanceseg.train.trainer_exporter import TrainerExporter


def get_trainer_exporter(out_dir, cfg):
    tensorboard_writer = SummaryWriter(log_dir=out_dir)

    exporter = TrainerExporter(out_dir, tensorboard_writer=tensorboard_writer, export_activations=cfg[
        'export_activations'], activation_layers_to_export=cfg['activation_layers_to_export'],
                               write_instance_metrics=cfg['write_instance_metrics'])
    return exporter


def get_trainer(cfg, cuda, model, optim, dataloaders, problem_config, out_dir):
    exporter = get_trainer_exporter(out_dir, cfg)
    trainer = instanceseg.Trainer(cuda=cuda, model=model, optimizer=optim, train_loader=dataloaders['train'],
                                  val_loader=dataloaders['val'], max_iter=cfg['max_iteration'],
                                  instance_problem=problem_config, size_average=cfg['size_average'],
                                  interval_validate=cfg.get('interval_validate', len(dataloaders['train'])),
                                  loss_type=cfg['loss_type'], matching_loss=cfg['matching'], exporter=exporter,
                                  train_loader_for_val=dataloaders['train_for_val'],
                                  augment_input_with_semantic_masks=cfg['augment_semantic'],
                                  generate_new_synthetic_data_each_epoch=(
                                              cfg['dataset'] == 'synthetic' and cfg['infinite_synthetic']))
    exporter.link_to_trainer(trainer)
    return trainer
