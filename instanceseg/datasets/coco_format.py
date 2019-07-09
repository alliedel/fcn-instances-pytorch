from instanceseg.datasets import palettes
from instanceseg.utils.misc import AttrDict


def generate_default_rgb_color_list(n_semantic_classes):
    colors_list = palettes.create_label_rgb_list(n_semantic_classes)
    return colors_list


class CategoryCOCOFormat(AttrDict):
    def __init__(self, id, name, color, supercategory=None, isthing=True):
        self.id = id
        self.name = str(name)
        self.color = color
        self.supercategory = str(supercategory or name)
        self.isthing = isthing


def create_labels_table_from_list_of_labels(list_of_labels):
    labels_table = [CategoryCOCOFormat(**{'id': el.id,
                                          'name': el.name,
                                          'color': el.color,
                                          'supercategory': el.supercategory,
                                          'isthing': 1 if el.isthing else 0}) for el in list_of_labels]
    return labels_table


def create_default_labels_table_from_semantic_names(semantic_class_names, semantic_class_vals=None):
    semantic_class_vals = semantic_class_vals if semantic_class_vals is not None else list(range(semantic_class_names))
    colors = generate_default_rgb_color_list(semantic_class_names)
    labels_table = [CategoryCOCOFormat(**{'id': semantic_class_vals[i],
                                          'name': name,
                                          'color': colors[i],
                                          'supercategory': name,
                                          'isthing': 1 if name != 'background' else 0}) for i, name in
                    enumerate(semantic_class_names)]
    return labels_table
