# import the necessary packages
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
import dogs_vs_cats_config as config
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from simplepreprocessor import SimplePreprocessor
from patchpreprocessor import PatchPreprocessor
from meanpreprocessor import MeanPreprocessor
from trainingmonitor import TrainingMonitor
from hdf5datasetgenerator import HDF5DatasetGenerator
from alexnet import AlexNet
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import argparse
import json
import gc
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,
				help="path to *specific* model checkpoint to load")
args = vars(ap.parse_args())

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
						 width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
						 horizontal_flip=True, fill_mode="nearest")

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 128, aug=aug,
								preprocessors=[pp, mp, iap], classes=2)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 128,
							  preprocessors=[sp, mp, iap], classes=2)

# clean cache
del config.TRAIN_HDF5, config.VAL_HDF5
gc.collect()

# if there is no specific model checkpoint supplied, then initialize the network and compile the model
if args["model"] is None:
	print("[INFO] compiling model...")
	opt = Adam(lr=1e-3)
	model = AlexNet.build(width=227, height=227, depth=3,
						  classes=2, reg=0.0002)
	model.compile(loss="binary_crossentropy", optimizer=opt,
				  metrics=["accuracy"])

# otherwise, load the checkpoint from disk
else:
	print("[INFO] loading {}...".format(args["model"]))
	model = load_model(args["model"])

# construct the set of callbacks
path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(
	os.getpid())])
callbacks = [TrainingMonitor(path)]

# train the network
model.fit_generator(
	trainGen.generator(),
	steps_per_epoch=trainGen.numImages // 64,
	validation_data=valGen.generator(),
	validation_steps=valGen.numImages // 64,
	epochs=75,
	max_queue_size=16,
	callbacks=callbacks, verbose=1)

# save the model to file
print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)

# close the HDF5 datasets
trainGen.close()
valGen.close()


