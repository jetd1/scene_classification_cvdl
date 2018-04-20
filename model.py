import torch
import torch.nn as nn
import torchvision as tv

def get(checkpoint=None, cuda=True, parallel=False):
    densenet = tv.models.densenet161(pretrained=False)
    del densenet.classifier
    densenet.classifier = nn.Linear(in_features=2208, out_features=80)

    if checkpoint is not None:
        densenet.load_state_dict(torch.load(checkpoint))
    else:
        print('Warning: Using unintialized DenseNet')

    if parallel:
        densenet = nn.DataParallel(densenet)
 
    if cuda:
        densenet.cuda()

    return densenet

