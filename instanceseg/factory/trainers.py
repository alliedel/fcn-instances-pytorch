from tensorboardX import SummaryWriter

from instanceseg.train import metrics
import instanceseg
from instanceseg.train.trainer_exporter import TrainerExporter


def get_metric_makers(my_trainer):
    metric_maker_kwargs = {
        'problem_config': my_trainer.instance_problem,
        'component_loss_function': my_trainer.loss_fcn,
        'augment_function_img_sem': my_trainer.augment_image
        if my_trainer.augment_input_with_semantic_masks else None
    }
    metric_makers = {
        'val': metrics.InstanceMetrics(my_trainer.val_loader, **metric_maker_kwargs),
        'train_for_val': metrics.InstanceMetrics(my_trainer.train_loader_for_val, **metric_maker_kwargs)
    }
    return metric_makers


def get_trainer_exporter(cfg, cuda, model, optim, dataloaders, problem_config, out_dir):
    tensorboard_writer = SummaryWriter(log_dir=out_dir)
    exporter = TrainerExporter(out_dir, instance_problem=problem_config, tensorboard_writer=tensorboard_writer,
                               export_activations=cfg['export_activations'],
                               activation_layers_to_export=cfg['activation_layers_to_export'],
                               write_instance_metrics=cfg['write_instance_metrics'])
    return exporter


def get_trainer(cfg, cuda, model, optim, dataloaders, problem_config, out_dir):
    exporter = get_trainer_exporter(cfg, cuda, model, optim, dataloaders, problem_config, out_dir)
    trainer = instanceseg.Trainer(cuda=cuda, model=model, optimizer=optim, train_loader=dataloaders['train'],
                                  val_loader=dataloaders['val'], max_iter=cfg['max_iteration'],
                                  instance_problem=problem_config, size_average=cfg['size_average'],
                                  interval_validate=cfg.get('interval_validate', len(dataloaders['train'])),
                                  loss_type=cfg['loss_type'], matching_loss=cfg['matching'], exporter=exporter,
                                  train_loader_for_val=dataloaders['train_for_val'],
                                  augment_input_with_semantic_masks=cfg['augment_semantic'],
                                  generate_new_synthetic_data_each_epoch=(
                                          cfg['dataset'] == 'synthetic' and cfg['infinite_synthetic']))
    metric_makers = get_metric_makers(trainer)
    trainer.exporter.add_metric_makers(metric_makers)
    return trainer
