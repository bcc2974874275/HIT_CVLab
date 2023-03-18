

'''
# --------------------------------------------
# select dataset
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# --------------------------------------------
'''


def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()

    # -----------------------------------------
    # super-resolution
    # -----------------------------------------
    if dataset_type in ['sr', 'super-resolution']:
        from data.dataset_sr import DatasetSR as D


    elif dataset_type in ['bsrnet', 'bsrgan', 'blindsr']:
        from data.dataset_blindsr import DatasetBlindSR as D

    elif dataset_type in ['pansharpening']:
        from data.dataset_pansharpening import DatasetPansharpening as D



    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
