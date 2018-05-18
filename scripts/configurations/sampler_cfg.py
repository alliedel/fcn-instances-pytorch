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
        'train_for_val': 'copy_train'
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
    }
}
