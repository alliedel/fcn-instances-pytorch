import os

import instanceseg.datasets.synthetic
import instanceseg.datasets.voc
import scripts.configurations.synthetic_cfg
import scripts.configurations.voc_cfg
from instanceseg.datasets import cityscapes, voc
from instanceseg.datasets import dataset_generator_registry
from scripts.configurations import cityscapes_cfg

PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, '..', '..')))
CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


class RegisteredDataset(object):
    def __init__(self, name, default_config, config_option_dict, dataset_generator,
                 dataset_path='/tmp/', original_semantic_class_names=None):
        self.name = name
        self.default_config = default_config
        self.config_options = config_option_dict
        self.original_semantic_class_names = original_semantic_class_names
        self.dataset_path = dataset_path
        self.dataset_generator = dataset_generator

    @property
    def cache_path(self):
        path = os.path.join(CACHE_DIR, self.name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_instance_count_filename(self, split, transformer_tag):
        return os.path.join(self.cache_path,
                            '{}_instance_counts_{}.npy'.format(split, transformer_tag))

    def get_semantic_pixel_count_filename(self, split, transformer_tag):
        return os.path.join(self.cache_path, '{}_semantic_pixel_counts_{}.npy'.format(
            split, transformer_tag))

    def get_occlusion_counts_filename(self, split, transformer_tag):
        return os.path.join(self.cache_path, '{}_within_class_occlusion_counts_{}.npy'.format(
            split, transformer_tag))


def get_cityscapes_labels_table(semantic_vals, semantic_names):

    categories = []
    for semantic_val, semantic_class_name in zip(semantic_vals, semantic_names):
        el = cityscapes.labels_table[semantic_class_name]
        categories.append({'id': el.id,
                           'name': el.name,
                           'color': el.color,
                           'supercategory': el.category,
                           'isthing': 1 if el.hasInstances else 0})
    return categories


REGISTRY = {
    'cityscapes': RegisteredDataset(
        name='cityscapes',
        default_config=cityscapes_cfg.get_default_train_config(),
        config_option_dict=cityscapes_cfg.train_configurations,
        original_semantic_class_names=cityscapes.RawCityscapesBase.get_semantic_class_names(),
        dataset_path=cityscapes.get_default_cityscapes_root(),
        dataset_generator=lambda cfg, split: dataset_generator_registry.get_dataset('cityscapes', cfg, split,
                                                                                    transform=True),
    ),
    'voc': RegisteredDataset(
        name='voc',
        default_config=scripts.configurations.voc_cfg.get_default_config(),
        config_option_dict=scripts.configurations.voc_cfg.configurations,
        original_semantic_class_names=instanceseg.datasets.voc.ALL_VOC_CLASS_NAMES,
        dataset_path=voc.get_default_voc_root(),
        dataset_generator=lambda cfg, split: dataset_generator_registry.get_dataset('voc', cfg, split,
                                                                                    transform=True)
    ),
    'synthetic': RegisteredDataset(
        name='synthetic',
        default_config=scripts.configurations.synthetic_cfg.get_default_train_config(),
        config_option_dict=scripts.configurations.synthetic_cfg.train_configurations,
        original_semantic_class_names=instanceseg.datasets.synthetic.ALL_BLOB_CLASS_NAMES,
        dataset_path='/tmp/',
        dataset_generator=lambda cfg, split: dataset_generator_registry.get_dataset('synthetic', cfg, split,
                                                                                    transform=True)
    )
}

