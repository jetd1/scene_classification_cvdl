import torch
import torchvision as tv
from torch.autograd import Variable as var
import argparse
import PIL.Image as im
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--single', '-s')
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--parallel', '-p', action='store_true')
parser.add_argument('--tencrop', '-t', action='store_true')
parser.add_argument('--input_dir', '-i')
parser.add_argument('--output_file', '-o')
parser.add_argument('--checkpoint', '-c')

args = parser.parse_args()

normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

def RunSingleTest(net):
    img = im.open(args.single)
    if args.tencrop:
        img = img.resize((256, 256), im.LANCZOS)
        img = tv.transforms.TenCrop((224, 224))(img)
        img = torch.stack([normalize(tv.transforms.ToTensor()(crop)) for crop in img])
    else:
        img = img.resize((224, 224), im.LANCZOS)
        img = normalize(tv.transforms.ToTensor()(img))

    img = var(img.view(1, *img.shape))
    
    if not args.cpu:
        img.cuda()

    if args.tencrop:
        bs, ncrops, c, h, w = img.size()
        img = img.view(-1, c, h, w)
   
    ret = net(img)

    if args.tencrop:
        ret = ret.view(bs, ncrops, -1).mean(1)
    
    _, p3 = torch.topk(ret, 3)
    p3 = p3.view(-1).cpu().data.numpy()
    
    print('Top-3 predictions:')
    for i in range(3):
        print(utils.idx2name(p3[i]), float(ret[0, p3[i]]), sep='\t')


def RunWholeTest(net):
    test_set = utils.NamedImageDataset(args.input_dir)
    if args.tencrop:
        test_set.transform = tv.transforms.Compose([
            tv.transforms.Resize((256, 256), im.LANCZOS),
            tv.transforms.TenCrop((224, 224)),
            tv.transforms.Lambda(lambda crops: torch.stack([normalize(tv.transforms.ToTensor()(crop)) for crop in crops])),
        ])
    else:
        test_set.transform = tv.transforms.Compose([
            tv.transforms.Resize((224, 224), im.LANCZOS),
            tv.transforms.ToTensor(),
            normalize
        ])

    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1
    )
    
    utils.WritePred(net, test_set_loader, args.output_file, not args.cpu, args.tencrop)

if __name__ == '__main__':
    import model
    densenet = model.get(args.checkpoint, not args.cpu, args.parallel)

    # !!! Crucial
    densenet.eval()

    if args.single is not None:
        RunSingleTest(densenet)
    else:
        RunWholeTest(densenet)
