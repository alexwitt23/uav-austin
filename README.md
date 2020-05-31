
## Setup

Make sure to have Python3 installed. If you would like to use a gpu, install 
CUDA and CUDNN. Next, run:
```
pip install -U Cython==0.29.15 numpy==1.17.4
pip install -r requirements.txt
```
Then install either `requirements-cpu.txt` or `requirements-gpu.txt` depending 
on your device.

## Data Generation

To create data to train an object detector on, run:
```
PYTHONPATH=$(pwd) data_generation/create_detection_data.py 
```

To create data to train a classifier, run:
```
PYTHONPATH=$(pwd) data_generation/create_clf_data.py 
```
Edit the `data_generation/config.yaml` to adjust the amount of data to create.


## Classifier Training

See example model configurations in `models/configs` for inspiration on a model 
architecture. Check out `torchvision` for all the possibilities. An example 
training command is:
```
PYTHONPATH=$(pwd) train/train_clf.py \
    --model_config configs/resnet18.yaml 
```

## Detector Training 

See example model configurations in `models/configs` for inspiration on a model 
architecture.
```
PYTHONPATH=$(pwd) train/train_det.py \
    --model_config configs/vovnet-det.yaml
```

## Inference on an Image

Inference on an image will utilize both the classifier and object detector model. 
The inference script can be run as follows:
```
PYTHONPATH=$(pwd) inference/find_targets.py \
    --image_path /path/to/image \
    --visualization_dir /path/to/save
```
The command above will visualize and save the models' predictions. See 
`inference/find_targets.py` for the full list of available arguments.


## Testing

Testing is performed by running `tox` at the repo root. The test in this repo consist
of python unittests and doctests.

#### Other

Updating submodules: `git submodule update --remote --merge`

# TODO

- [ ] Devise a way to be able to create detection, classification, and feature data in 
separate independently. 

- [ ] Finalize training pipelines.

- [ ] Finish inference pipeline.

- [ ] Work on Box Dev integration if platform available. 

- [ ] Weights & Biases integration once project is granted.

- [ ] GitHub Actions CI and unittests
