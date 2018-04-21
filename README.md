## Requirements

python >= 3.5, with

PIL,

tqdm,

pytorch,

torchvision.



## Testing

There's already a trained weight file under `./checkpoints/`, it reaches accuracy of 83.2% and top-3 accuracy of 95.4%.

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
#filename       #Top-3 classes
test/051247.jpg 15 11 4
test/012087.jpg 49 58 50
...
```

#### Addictional Options

```
--cpu             Test on CPUs rather than GPUs
--parallel, -p    Run batches parallel (using nn.DataParallel)
--tenctop, -t     Run tencrop test
```



## Training

Will be release after HW01 submission



