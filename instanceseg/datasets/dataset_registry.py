# cityscapes
import scripts.configurations.cityscapes_cfg
# synthetic
import scripts.configurations.synthetic_cfg
# voc
import scripts.configurations.voc_cfg
import instanceseg.datasets.cityscapes
import instanceseg.datasets.synthetic
import instanceseg.datasets.voc
from instanceseg.datasets import dataset_generator_registry


class RegisteredDataset(object):
    def __init__(self, name, default_config, config_option_dict, dataset_generator, dataset_path='/tmp/',
                 original_semantic_class_names=None):
        self.name = name
        self.default_config = default_config
        self.config_options = config_option_dict
        self.original_semantic_class_names = original_semantic_class_names
        self.dataset_path = dataset_path
        self.dataset_generator = dataset_generator


REGISTRY = {
    'cityscapes': RegisteredDataset(
        name='cityscapes',
        default_config=scripts.configurations.cityscapes_cfg.get_default_config(),
        config_option_dict=scripts.configurations.cityscapes_cfg.configurations,
        original_semantic_class_names=instanceseg.datasets.cityscapes.RawCityscapesBase.get_semantic_class_names(),
        dataset_path=instanceseg.datasets.cityscapes.get_default_cityscapes_root(),
        dataset_generator=lambda cfg: dataset_generator_registry.get_dataset('cityscapes', cfg, transform=True),
    ),
    'voc': RegisteredDataset(
        name='voc',
        default_config=scripts.configurations.voc_cfg.get_default_config(),
        config_option_dict=scripts.configurations.voc_cfg.configurations,
        original_semantic_class_names=instanceseg.datasets.voc.ALL_VOC_CLASS_NAMES,
        dataset_path=instanceseg.datasets.voc.get_default_voc_root(),
        dataset_generator=lambda cfg: dataset_generator_registry.get_dataset('voc', cfg, transform=True)
    ),
    'synthetic': RegisteredDataset(
        name='synthetic',
        default_config=scripts.configurations.synthetic_cfg.get_default_config(),
        config_option_dict=scripts.configurations.synthetic_cfg.configurations,
        original_semantic_class_names=instanceseg.datasets.synthetic.ALL_BLOB_CLASS_NAMES,
        dataset_path='/tmp/',
        dataset_generator=lambda cfg: dataset_generator_registry.get_dataset('synthetic', cfg, transform=True)
    )
}
