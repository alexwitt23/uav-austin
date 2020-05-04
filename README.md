# TODO
* Creating classification data shouldn't depend on making detection data first.
Can these be made independent with shared utilities?

* Finalize detector training

* Finish inference pipeline

## Setup
Make sure to have Python3 installed. If you would like to use a gpu, install 
CUDA and CUDNN. Next, run:
```
pip install -U Cython==0.29.15 numpy==1.17.4
pip install -r requirements.txt
```
Then install either ``requirements-cpu.txt or `requirements-gpu.txt` depending 
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
    --model_config models/configs/resnet18.yaml 
```

## Testing
Testing is performed by running `tox` at the repo root. The test in this repo consist of python unittests and doctests.