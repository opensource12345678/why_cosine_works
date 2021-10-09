# Eigencurve: Optimal Learning Rate Schedule for SGD on Quadratic Objectives with Skewed Hessian Spectrums

## Dependency

All code are implemented in Python. One can install the dependency as follows,

```sh
pip install -r requirements.txt
```

## ImageNet

All codes are stored in branch "imagenet", please checkout to that
branch first.

```sh
git checkout imagenet
```

An alternative is to download the code of that branch to run this part of the
experiments.

```
https://github.com/opensource12345678/why_cosine_works/tree/imagenet
```

### ----- Data Preparation

#### Download ImageNet Data

Before running this part of experiments, one has to download data from
[ImageNet ILSVRC2012](https://image-net.org/challenges/LSVRC/2012/) first.
Notice that successful download requires registrating an account in that
website and verifying the institute after registration.

After obtaining a verified account, one can download the [Training
set](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar) and
[Validation
set](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar). The test
set is not used, which is the same setting as in most papers, since the labels
are not released in that website.

#### Preprocess ImageNet Data

The downloaded data has to be uncompressed, preprocessed and put under
`data/ImageNet` in following directory structures/formats,

```sh
data/ImageNet/
|- train/
  |- n01440764
    |- n01440764_10026.JPEG
    |- n01440764_10027.JPEG
    |- n01440764_10029.JPEG
    |- n01440764_10040.JPEG
    ...
    |- n01440764_9981.JPEG
  |- n01443537
  ...
  |- n15075141
|- val/
  |- n01440764
    |- ILSVRC2012_val_00000293.JPEG
    |- ILSVRC2012_val_00002138.JPEG
    ...
    |- ILSVRC2012_val_00048969.JPEG
  |- n01443537
  ...
  |- n15075141
```

For those who are not familiar with this process, please follow the next
instructions step by step,

**Step 1:** Put the download `ILSVRC2012_img_train.tar` and
`ILSVRC2012_img_val.tar` under certain directory.

```sh
# Assumes that you are under the main directory `why_cosine_works/`
mkdir -p data/ImageNet/
mv ILSVRC2012_img_train.tar data/ImageNet/
mv ILSVRC2012_img_val.tar data/ImageNet/
cd data/ImageNet/

# Uncompresses those files
tar -xvf ILSVRC2012_img_train.tar -C train
tar -xvf ILSVRC2012_img_val.tar -C val

# After uncompression, you will see a bunch of ".tar" files under `train/` and
# a bunch of ".JPEG" files under `val/`
```

**Step 2:** Prepare the training set. We have included a script `extract.sh`
under `data/ImageNet/` to facilitate this process.

```sh
# Assumes that you are currently under the directory `why_cosine_works/data/ImageNet/`
chmod +x extract.sh
cd train
for tar in *.tar;  do sh ../extract.sh $tar; done

# The second command takes about 1h to finish. After it completes, the "train/"
# should be organized with the desired structured.
cd ..
```

**Step 3:** Prepare the validation set. This task majorly involves putting
".JPEG" files in directories named with their corresponding labels.
Fortunately, thanks to soumith, we have an available a script for this tedious
task. One may refer to [the
script](https://github.com/soumith/imagenetloader.torch/blob/master/valprep.sh)
for the content if interested.

```sh
# Assumes that you are currently under the directory `why_cosine_works/data/ImageNet/`

# Obtains the script for preprocessing validation set via wget
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

# Runs the script
chmod +x valprep.sh
cd val
../valprep.sh

# The script should take less than 10mins to finish. After it completes, the
# "val/" should be oragnized with the disired structure.
```

### ----- Training and Results Collection

With prepared data, two scripts are used for generating the imagenet results.
`run_imagenet.sh` trains the resnet-18 model with different cosine-power
scheduler, which assumes the existence of at least two GPU cards.

After it finished, run `collect_results.sh` to collect info from log files.

```sh
./run_imagenet.sh
# Should take at least three weeks in two NVIDIA GeForce 2080 Ti cards

./collect_results.sh
# Should take less than 1mins
```
