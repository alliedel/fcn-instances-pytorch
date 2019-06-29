from instanceseg.datasets import instance_dataset
from instanceseg.utils.instance_utils import InstanceProblemConfig


class CategoryCOCOFormat(object):
    def __init__(self, id, name, color, supercategory, isthing):
        self.id = id
        self.name = name
        self.color = color
        self.supercategory = supercategory
        self.isthing = isthing


def create_labels_table(list_of_labels):
    labels_table = [{'id': el.id,
                     'name': el.name,
                     'color': el.color,
                     'supercategory': el.supercategory,
                     'isthing': 1 if el.isthing else 0} for el in list_of_labels]
    return labels_table


