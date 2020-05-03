# TODO

## Classifier Training
See example model configurations in `models/configs` for inspiration on a model architecture. Check out `torchvision` for
all the possibilities. An example training command is:
```
PYTHONPATH=$(pwd) train/train_clf.py \
    --model_config models/configs/resnet18.yaml 
```