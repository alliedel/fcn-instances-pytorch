# TODO(allie): Make it easier to generate and override sampler_cfgs (like it is to generate the others)


sampler_cfgs = {
    'default': {
        'train':
            {
                'n_images': None,
                'sem_cls_filter': None,
                'n_instances_range': None,
            },
        'val':
            {
                'n_images': None,
                'sem_cls_filter': None,
                'n_instances_range': None,
            },
        'train_for_val':  # just configures what should be processed during val
            {
                'n_images': None  # Change to reduce amount of images used to 'validate' the training set
            }
    },
    'person_2inst_1img': {
        'train':
            {'n_images': 1,
             'sem_cls_filter': ['person'],
             'n_instances_range': (2, None),
             },
        'val': 'copy_train',
        'train_for_val':
            {
                'n_images': None
            }
    },
    'person_2inst_2img': {
        'train':
            {'n_images': 2,
             'sem_cls_filter': ['person'],
             'n_instances_range': (2, None),
             },
        'val': 'copy_train'
    },
    'person_2inst_allimg_sameval': {
        'train':
            {'n_images': None,
             'sem_cls_filter': ['person'],
             'n_instances_range': (2, None),
             },
        'val': 'copy_train'
    },
    'person_2inst_allimg_realval': {
        'train':
            {'n_images': None,
             'sem_cls_filter': ['person'],
             'n_instances_range': (2, None),
             },
        'val':
            {'n_images': None,
             'sem_cls_filter': ['person'],
             'n_instances_range': (2, None),
             },
        'train_for_val':
            {
                'n_images': None
            }
    },
    'person_2inst_20img_sameval': {
        'train':
            {'n_images': 20,
             'sem_cls_filter': ['person'],
             'n_instances_range': (2, None),
             },
        'val': 'copy_train',
        'train_for_val':  {
            'n_images': None
        }
    },
    'person_2_4inst_allimg_realval': {
        'train':
            {'n_images': None,
             'sem_cls_filter': ['person'],
             'n_instances_range': (2, 4),
             },
        'val':
            {'n_images': None,
             'sem_cls_filter': ['person'],
             'n_instances_range': (2, 4),
             },
        'train_for_val':  # just configures what should be processed during val
            {
                'n_images': None
            }
    },
    'person_2_4inst_1img': {
        'train':
            {'n_images': 1,
             'sem_cls_filter': ['person'],
             'n_instances_range': (2, 4),
             },
        'val': 'copy_train',
        'train_for_val':  {
            'n_images': None
        }
    },
    'car_2_4inst_allimg_realval': {
        'train':
            {'n_images': None,
             'sem_cls_filter': ['car'],
             'n_instances_range': (2, 4),
             },
        'val':
            {'n_images': None,
             'sem_cls_filter': ['car'],
             'n_instances_range': (2, 4),
             },
        'train_for_val':  # just configures what should be processed during val
            {
                'n_images': None
            }
    },
    'car_2inst_allimg_realval': {
        'train':
            {'n_images': None,
             'sem_cls_filter': ['car'],
             'n_instances_range': (2, None),
             },
        'val':
            {'n_images': None,
             'sem_cls_filter': ['car'],
             'n_instances_range': (2, None),
             },
        'train_for_val':  # just configures what should be processed during val
            {
                'n_images': None
            }
    },
    'car_2_5inst_allimg_realval': {
        'train':
            {'n_images': None,
             'sem_cls_filter': ['car'],
             'n_instances_range': (2, 5 + 1),
             },
        'val':
            {'n_images': None,
             'sem_cls_filter': ['car'],
             'n_instances_range': (2, 5 + 1),
             },
        'train_for_val':  # just configures what should be processed during val
            {
                'n_images': None
            }
    },
    'car_2_3': {
        'train':
            {'n_images': None,
             'sem_cls_filter': ['car'],
             'n_instances_range': (2, 3 + 1),
             },
        'val':
            {'n_images': None,
             'sem_cls_filter': ['car'],
             'n_instances_range': (2, 3 + 1),
             },
        'train_for_val':  # just configures what should be processed during val
            {
                'n_images': None
            }
    }
}

sampler_cfgs[None] = sampler_cfgs['default']
