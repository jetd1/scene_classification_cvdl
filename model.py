# Xiaoshuai 'Jet' Zhang, jet@pku.edu.cn
# HW01 for CVDL course of PKU

import torch
import torch.nn as nn
import torchvision as tv

def getDenseNet(checkpoint, cuda=True, parallel=False):
    densenet = tv.models.densenet161(pretrained=False)
    del densenet.classifier
    densenet.classifier = nn.Linear(in_features=2208, out_features=80)

    if checkpoint == './checkpoints/place80.pth':
        print('Using pretrained DenseNet...')
    state_dict = torch.load(checkpoint)

    # Hack for newer pytorch and torchvision
    try:
        densenet.load_state_dict(state_dict)
    except:
        import collections
        nstate_dict = collections.OrderedDict()
        for k in state_dict:
            if k.count('.') > 3:
                idx = k[:k.rfind('.')].rfind('.')
                nk = k[:idx] + k[idx+1:]
                nstate_dict[nk] = state_dict[k]
        densenet.load_state_dict(nstate_dict)

    if parallel:
        densenet = nn.DataParallel(densenet)
 
    if cuda:
        densenet.cuda()

    return densenet

