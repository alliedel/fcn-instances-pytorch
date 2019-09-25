from instanceseg.datasets import cityscapes
from . import generic_cfg


def get_default_train_config():
    default_cfg = generic_cfg.get_default_train_config()
    default_cfg.update(
        dict(n_instances_per_class=3,
             set_extras_to_void=True,
             lr=1.0e-5,
             dataset='cityscapes',
             resize=True,
             resize_size=(281, 500),  # (512, 1024)
             dataset_path=cityscapes.get_default_cityscapes_root()
             )
    )
    return default_cfg


train_configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    0: dict(),
    'person_only__semantic': dict(
        semantic_subset=['person', 'background'],
        set_extras_to_void=True,
        single_instance=True,
        n_instances_per_class=None,
    ),
    'car_only__single_instance_semantic': dict(
        semantic_subset=['car', 'background'],
        interval_validate=1000,
        max_iteration=100000,
        n_instances_per_class=1,
        single_instance=True,
        map_to_semantic=True,
        dataset_instance_cap=None,
    ),
    'car_only__mapped_semantic': dict(
        n_instances_per_class=1,
        max_iteration=1000000,
        single_instance=True,
        interval_validate=4000,
        semantic_subset=['car', 'background'],
        map_to_semantic=True,
        dataset_instance_cap=None,
    ),
    'semantic': dict(
        single_instance=True,
        n_instances_per_class=1,
        dataset_instance_cap=None,
        map_to_semantic=True,       # TODO(allie): fix the fact that we have to do this.
    ),
    'semantic_bypass_match': dict(
        single_instance=True,
        n_instances_per_class=1,
        dataset_instance_cap=None,
        map_to_semantic=True,       # TODO(allie): fix the fact that we have to do this.
        matching=False
    ),
    'semantic_v2': dict(
        dataset_instance_cap=None,
        map_to_semantic=True,       # TODO(allie): fix the fact that we have to do this.
    ),
    'objectsemantic': dict(
        single_instance=True,
        n_instances_per_class=1,
        dataset_instance_cap=None,
        semantic_subset=[class_name for class_name, has_instances in zip(
            cityscapes.labels_table_cityscapes.class_names, cityscapes.labels_table_cityscapes.has_instances)
                         if has_instances]
    ),
    'car_instance': dict(
        max_iteration=1000000,
        interval_validate=4000,
        n_instances_per_class=6,
        dataset_instance_cap='match_model',
        ordering=None,
        semantic_subset=['car', 'background'],
    ),
    'car_person_instance': dict(
        max_iteration=1000000,
        interval_validate=4000,
        n_instances_per_class=4,
        dataset_instance_cap='match_model',
        ordering=None,
        semantic_subset=['car', 'person', 'background'],
    ),
    'vehicle_instance': dict(
        max_iteration=1000000,
        interval_validate=4000,
        n_instances_per_class=3,
        dataset_instance_cap='match_model',
        ordering=None,
        semantic_subset=['car', 'bus', 'train'],
    ),
    'debug_dataloader_only': dict(
        debug_dataloader_only=True,
        sampler='overfit_1_car_person',
        semantic_subset=['car', 'person'],
),
    'debug_dataloader_only_10_imgs': dict(
        debug_dataloader_only=True,
        sampler='overfit_10_car_person',
        semantic_subset=['car', 'person'],
    ),
    'overfit_10': dict(
        sampler='overfit_10_car_person',
        semantic_subset=['car', 'person'],
        n_instances_per_class=3,
        dataset_instance_cap='match_model',
        interval_validate=100
    ),
    'overfit_1': dict(
        sampler='overfit_1_car_person',
        semantic_subset=['car', 'person'],
        n_instances_per_class=3,
        dataset_instance_cap='match_model',
        max_iteration=10000,
        interval_validate=20
    ),
    'resnet_car_2_4': dict(
        max_iteration=1000000,
        sampler='car_2_4',
        interval_validate=4000,
        n_instances_per_class=4,
        dataset_instance_cap='match_model',
        ordering=None,
        semantic_subset=['car'],
        backbone='resnet50'
    ),
    'resnet_car_2_5': dict(
        max_iteration=1000000,
        sampler='car_2_5',
        interval_validate=4000,
        n_instances_per_class=5,
        dataset_instance_cap='match_model',
        ordering=None,
        semantic_subset=['car'],
        backbone='resnet50'
    ),
    'car_2_4_plus_semantics': dict(
        max_iteration=1000000,
        sampler='car_2_4',
        interval_validate=4000,
        n_instances_per_class=4,
        dataset_instance_cap='match_model',
        ordering=None,
        semantic_subset=['car', 'sky', 'vegetation', 'sidewalk', 'building', 'road', 'wall'],
        backbone='resnet50',
        ),
    'full_panoptic_5each': dict(
        max_iteration=10000000,
        sampler=None,
        interval_validate=4000,
        n_instances_per_class=5,
        dataset_instance_cap=None,
        ordering=None,
        semantic_subset=None,
        backbone='resnet50',
        ),
    # 'debug_all': dict(
    #     max_iteration=10000,
    #     interval_validate=100,
    #     n_instances_per_class=3,
    #     dataset_instance_cap='match_model',
    #     ordering=None,
    #     sampler='overfit_1_car',
    #     debug_dataloader=True
    # ),
}
