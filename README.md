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




