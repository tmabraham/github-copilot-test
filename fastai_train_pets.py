"""
1. Import the fastai v2 library
2. Load the Oxford Pets dataset from URLs.PETS
3. Create a DataLoaders object with the path and filenames. Here, the label for the image can be provided by a regular expression pattern that, when given a filename string, gives the text before the underscore, number, and filename extension. For example, if the filename is "great_pyrenees_173.jpg" the regular expression returns "great_pyrenees"
4. Create a Learner with cnn_learner, using an Imagenet-pretrained ResNet50
5. Finetune the model for 10 epochs.
"""

# 1. Import the fastai v2 library
from fastai.vision import *

# 2. Load the Oxford Pets dataset from URLs.PETS
path = untar_data(URLs.PETS)

# 3. Create a DataLoaders object with the path and filenames. Here, the label for the image can be provided by a regular expression pattern that, when given a filename string, gives the text before the underscore, number, and filename extension. For example, if the filename is "great_pyrenees_173.jpg" the regular expression returns "great_pyrenees"
data = ImageDataBunch.from_folder(path, valid_pct=0.2, ds_tfms=get_transforms(), size=224, bs=64)

# 4. Create a Learner with cnn_learner, using an Imagenet-pretrained ResNet50
learn = cnn_learner(data, models.resnet50, metrics=error_rate)

# 5. Finetune the model for 10 epochs.
learn.fit_one_cycle(10)
