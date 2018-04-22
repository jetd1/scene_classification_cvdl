A deep-based model for 80-class scene classification. This is HW01 for CVDL course of PKU. The report is at `./report`. You can test online [here](http://s.buriedjet.com/cvdl1/).

This repo uses Git LFS in order to upload checkpoints. So you should clone using:

```c
git lfs clone https://github.com/BuriedJet/scene_classification_cvdl.git
```

rather than normal `git clone`. You may need to install [Git LFS](https://git-lfs.github.com/) first (or the weights can be downloaded from [here](http://s.buriedjet.com/filehost/place80.pth))

## Dependencies

python (>= 3.5), with

- PIL (>= 4.1.1),
- tqdm (>= 4.23.0),
- pytorch (>= 0.3.0),
- torchvision (>= 0.2.0).

## Testing

A trained weight file is under `./checkpoints`, which reaches 83.2% acc and 95.4% top-3 acc on validation set.

#### Single Image Testing

To perform classification on an image:

```bash
python test.py -c [CHECKPOINT_PATH] -s [IMAGE_PATH]
```

#### Image Directory Testing

To perform classification on an image directory:

```bash
python test.py -c [CHECKPOINT_PATH] -i [IMAGE_DIR] -o [OUTPUT_FILE]
```

The output file will be in the format of:

```
# Filename      # Top-3 classes
test/051247.jpg 15 11 4
test/012087.jpg 49 58 50
...
```

#### Addictional Options

```
--cpu             Test on CPUs rather than on GPUs
--parallel, -p    Run batches parallel (using nn.DataParallel)
--tenctop, -t     Run tencrop test
```



## Training

Will be released after HW01 submission


