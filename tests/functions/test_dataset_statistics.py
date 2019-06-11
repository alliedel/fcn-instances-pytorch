import os
import torch

from instanceseg.datasets import dataset_statistics
from instanceseg.datasets import cityscapes

UNITTEST_CITYSCAPES_DATASET_ROOT = './tests/test_data/cityscapesunittest/'

# noinspection PyArgumentList
OCCLUSION_COUNT_GT = {
    'img_basenames': ('aachen_000000_000019_leftImg8bit.png',
                      'aachen_000001_000019_leftImg8bit.png',
                      'aachen_000002_000019_leftImg8bit.png',
                      'aachen_000003_000019_leftImg8bit.png',
                      'aachen_000004_000019_leftImg8bit.png',
                      'aachen_000005_000019_leftImg8bit.png',
                      'aachen_000006_000019_leftImg8bit.png',
                      'aachen_000007_000019_leftImg8bit.png',
                      'aachen_000008_000019_leftImg8bit.png',
                      'aachen_000009_000019_leftImg8bit.png'
                      ),
    'occlusion_counts': (
        torch.IntTensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 9, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 1, 0, 0, 0, 0, 0]
            ])

    )
}


def is_dataset_for_stored_occlusions(unittest_cityscapes_dataset):
    unittest_basenames = tuple(os.path.basename(f['img']) for f in
                               unittest_cityscapes_dataset.raw_dataset.files)
    if unittest_basenames != OCCLUSION_COUNT_GT['img_basenames']:
        return False
    # if
    #     return
    return True


def get_unittest_cityscapes_dataset(root=UNITTEST_CITYSCAPES_DATASET_ROOT):
    unittest_cityscapes = cityscapes.TransformedCityscapes(root=root,
                                                           split='train')
    return unittest_cityscapes


def test_occlusions_on_select_cityscapes_car_images():
    debug = True
    unittest_cityscapes_dataset = get_unittest_cityscapes_dataset()
    assert is_dataset_for_stored_occlusions(unittest_cityscapes_dataset)
    occlusion_cache = dataset_statistics.OcclusionsOfSameClass(
        range(len(unittest_cityscapes_dataset.semantic_class_names)),
        semantic_class_names=unittest_cityscapes_dataset.semantic_class_names,
        cache_file=None, compute_batch_size=1, debug=debug)
    # occlusion_cache.compute_or_retrieve(unittest_cityscapes_dataset)
    occlusion_counts = occlusion_cache.compute_occlusion_counts(
        unittest_cityscapes_dataset,
        occlusion_cache.semantic_class_vals,
        compute_batch_size=occlusion_cache.compute_batch_size)
    occlusion_cache.pprint_stat_tensor(stat_tensor=occlusion_counts,
                                       labels=occlusion_cache.labels, with_labels=True)
    assert occlusion_counts.equal(OCCLUSION_COUNT_GT['occlusion_counts'])
    return occlusion_counts


def test_occlusion_finder():
    unittest_cityscapes_dataset = get_unittest_cityscapes_dataset()
    i, (sl, il) = unittest_cityscapes_dataset[0]
    n_occlusion_pairings_per_sem_cls, arr_of_occlusion_locations_per_cls = \
        dataset_statistics.get_occlusions_hws_from_labels(
            sl, il, range(unittest_cityscapes_dataset.n_semantic_classes))
    test_export_dir = '/tmp/test_occlusion_finder'
    print(arr_of_occlusion_locations_per_cls.shape)
    dataset_statistics.OcclusionsOfSameClass.clear_and_create_dir(test_export_dir)
    for sem_ch in range(arr_of_occlusion_locations_per_cls.shape[2]):
        debug_img = arr_of_occlusion_locations_per_cls[:, :, sem_ch] + (sl == sem_ch).astype(
            'uint8')
        basename = 'occlusion_single_img_sem_cls_{}'.format(
            unittest_cityscapes_dataset.semantic_class_names[sem_ch])
        full_img_name = os.path.join(test_export_dir, basename + '.png')
        dataset_statistics.OcclusionsOfSameClass.export_label_image(debug_img, full_img_name)
        print(full_img_name)
    print('Test images written to {}'.format(test_export_dir))


if __name__ == '__main__':
    occlusion_cache = test_occlusions_on_select_cityscapes_car_images()
    test_occlusion_finder()
