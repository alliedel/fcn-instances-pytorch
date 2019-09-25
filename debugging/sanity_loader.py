from instanceseg.datasets import dataset_generator_registry


def main():
    datasets, transformer_tag = dataset_generator_registry.get_default_datasets_for_instance_counts('cityscapes',
                                                                                                    splits=(
                                                                                                    'train', 'val'))
    dataset = datasets['train']
    datapoint = dataset[0]

    print('id: ', datapoint['identifier'])
    image, sem_lbl, inst_lbl = datapoint['image'], datapoint['sem_lbl'], datapoint['inst_lbl']

    dataset_length = len(dataset)

    id_list = dataset.id_list


if __name__ == '__main__':
    main()
