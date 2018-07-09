# cityscapes
import scripts.configurations.cityscapes_cfg
# synthetic
import scripts.configurations.synthetic_cfg
# voc
import scripts.configurations.voc_cfg
import torchfcn.datasets.cityscapes
import torchfcn.datasets.synthetic
import torchfcn.datasets.voc


class RegisteredDataset(object):
    def __init__(self, name, default_config, config_option_dict, dataset_path='/tmp/',
                 original_semantic_class_names=None):
        self.name = name
        self.default_config = default_config
        self.config_options = config_option_dict
        self.original_semantic_class_names = original_semantic_class_names
        self.dataset_path = dataset_path


REGISTRY = {
    'cityscapes': RegisteredDataset(
        name='cityscapes',
        default_config=scripts.configurations.cityscapes_cfg.get_default_config(),
        config_option_dict=scripts.configurations.cityscapes_cfg.configurations,
        original_semantic_class_names=torchfcn.datasets.cityscapes.RawCityscapesBase.semantic_class_names,
        dataset_path=torchfcn.datasets.cityscapes.get_default_cityscapes_root()
    ),
    'voc': RegisteredDataset(
        name='voc',
        default_config=scripts.configurations.voc_cfg.get_default_config(),
        config_option_dict=scripts.configurations.voc_cfg.configurations,
        original_semantic_class_names=torchfcn.datasets.voc.ALL_VOC_CLASS_NAMES,
        dataset_path=torchfcn.datasets.voc.get_default_voc_root()
    ),
    'synthetic': RegisteredDataset(
        name='synthetic',
        default_config=scripts.configurations.synthetic_cfg.get_default_config(),
        config_option_dict=scripts.configurations.synthetic_cfg.configurations,
        original_semantic_class_names=torchfcn.datasets.synthetic.ALL_BLOB_CLASS_NAMES,
        dataset_path='/tmp/'
    )
}
