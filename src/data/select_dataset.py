from src.typings.TrainOptions import Dataset


def get_dataset(dataset_opt: Dataset):  
    dataset_type = dataset_opt['dataset_type'].lower()
    if dataset_type in ['jpeg']:
        from .dataset_jpeg import DatasetJPEG as D

    elif dataset_type in ['jpeggray']:
        from .dataset_jpeggray import DatasetJPEG as D

    elif dataset_type in ['jpeggraydouble']:
        from .dataset_jpeggraydouble import DatasetJPEG as D

    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
