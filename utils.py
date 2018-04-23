# Xiaoshuai 'Jet' Zhang, jet@pku.edu.cn
# HW01 for CVDL course of PKU

import torch
import os
import sys
import csv
from torch import nn
from tqdm import tqdm
import PIL.Image as im
import torch.utils.data
from torch.autograd import Variable as var

_class_csv = None

def ensure_exists(dname):
    import os
    if not os.path.exists(dname):
        try:
            os.makedirs(dname)
        except:
            pass
    return dname

def ReadInfo(f):
    with open(f, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        return {row[0][row[0].rfind('/') + 1:]: int(row[1]) for row in reader}

def WritePred(net, t_loader, filename, cuda=True, tencrop=False, out_arr=None):
    assert t_loader.batch_size == 1
    with open(filename, 'w') as f:
        tl = len(t_loader)
        idx = 0
        for data in tqdm(t_loader, file=sys.stdout):
            in_img = var(data)
            if cuda:
                in_img = in_img.cuda()
            if tencrop:
                bs, ncrops, c, h, w = in_img.size()
                in_img = in_img.view(-1, c, h, w)                
            ret = net(in_img)
            if tencrop:
                ret = ret.view(bs, ncrops, -1).mean(1)
            _, p3 = torch.topk(ret, 3)
            p3 = p3.view(-1).cpu().data.numpy()
            print('test/' + t_loader.dataset.getName(idx), p3[0], p3[1], p3[2], file=f)
            if out_arr is not None:
                out_arr.append([p3[0], p3[1], p3[2]])
            idx += 1

def WriteSinglePred(ret, f, class_csv):
    _, p3 = torch.topk(ret, 3)
    p3 = p3.view(-1).cpu().data.numpy()
    ret = nn.functional.softmax(ret, dim=1)

    print('Top-3 predictions:', file=f)
    for i in range(3):
        print(idx2name(p3[i], class_csv), float(ret[0, p3[i]]), sep='\t', file=f)
    
def ReadJson(f):
    with open(f, 'r', newline='') as json_file:
        js_dict = json.load(json_file)
        return {i['image_id']: int(i['label_id']) for i in js_dict}

def idx2name(idx, class_csv):
    global _class_csv
    if _class_csv is None:
        with open(class_csv, 'r', encoding='utf8') as csv_file:
            reader = csv.reader(csv_file)
            _class_csv = {int(row[0]): row[1] for row in reader}
    return _class_csv[idx]

class NamedImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, label_dict=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = []
        self.name_list = []
        self.label_dict = label_dict

        for r, d, filenames in os.walk(self.root_dir):
            for f in filenames:
                if f[-3:] not in ['jpg', 'png']:
                    continue
                self.image_list.append(os.path.join(r, f))
                self.name_list.append(f)
        
        if label_dict is not None:
            self.labels = [None] * len(self.image_list)
            for idx in range(len(self.image_list)):
                self.labels[idx] = label_dict[self.name_list[idx]]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):            
        ret = im.open(self.image_list[idx])
        if self.transform:
            ret = self.transform(ret)
        if self.label_dict is not None:
            ret = (ret, self.labels[idx])
        return ret

    def getName(self, idx):
        return self.name_list[idx]


