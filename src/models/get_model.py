def get_model(opt):
    model = opt['model']      # one input: L

    if model == 'fbcnn':
        from .model_fbcnn import ModelFBCNN as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
