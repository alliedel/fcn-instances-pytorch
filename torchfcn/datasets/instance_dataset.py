from torch.utils import data
from . import dataset_utils


# original_dataset_class_names = ALL_VOC_CLASS_NAMES
# files = self.files[self.split]

class InstanceDataset(data.Dataset):

    def __init__(self, original_dataset_class_names, files, split):
        """

        :param original_dataset_class_names: list of class names for each value ['background', 'person', 'car', 'plane']
        :param files: list of dictionaries with fields ['img_file', 'sem_lbl_file', 'inst_lbl_file']
        :param split: 'train', 'val'
        """
        self.original_dataset_class_names = original_dataset_class_names
        self.n_inst_cap_per_class = None
        self.files = files
        self.split = split

        # Derived
        self.class_names = None
        self.idxs_int_all_names = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data_file = self.files[index]
        img, lbl = self.load_and_process_voc_files(img_file=data_file['img'],
                                                   sem_lbl_file=data_file['sem_lbl'],
                                                   inst_lbl_file=data_file['inst_lbl'])
        return img, lbl

    def set_instance_cap(self, n_inst_cap_per_class=None):
        if not isinstance(n_inst_cap_per_class, int):
            raise NotImplementedError('Haven\'t implemented dif cap per semantic class. Please use an int.')
        self.n_inst_cap_per_class = n_inst_cap_per_class

    def reset_instance_cap(self):
        self.n_inst_cap_per_class = None

    def reduce_to_semantic_subset(self, semantic_subset):
        self.class_names, self.idxs_into_all_names = dataset_utils.get_semantic_names_and_idxs(
            semantic_subset=semantic_subset, full_set=self.original_dataset_class_names)

    def clear_semantic_subset(self):
        self.class_names, self.idxs_into_all_names = dataset_utils.get_semantic_names_and_idxs(
            semantic_subset=None, full_set=self.original_dataset_class_names)


