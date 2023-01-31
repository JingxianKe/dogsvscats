from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-o", "--output", required=True,
                help="path to output HDF5 file")
ap.add_argument("-b", "--batch-size", type=int, default=16,
                help="batch size of images to be passed through network")
ap.add_argument("-s", "--buffer-size", type=int, default=1000,
                help="size of feature extraction buffer")
args = vars(ap.parse_args())

bs = args["batch_size"]

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)

labels = [p.split(os.path.sep)[-1].split(".")[0] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# 加载ResNet50网络
print("[INFO] loading network...")
model = ResNet50(weights="imagenet", include_top=False)

dataset = HDF5DatasetWriter((len(imagePaths), 2048 * 7 * 7),
                            args["output"], dataKey="features", bufSize=args["buffer_size"])
dataset.storeClassLabels(le.classes_)

# 显示进度条
widgets = ["Extracting Features: ", progressbar.Percentage(), " ",
           progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths),
                               widgets=widgets).start()

for i in np.arange(0, len(imagePaths), bs):
    batchPaths = imagePaths[i:i + bs]
    batchLabels = labels[i:i + bs]
    batchImages = []

    for (j, imagePath) in enumerate(batchPaths):
        # 加载输入图像，并确保其大小为224x224像素
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)

        # 预处理图像：增加一个维度；提取ImageNet数据集的RGB均值
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        batchImages.append(image)

    # 将图像传进网络，其输出作为我们想要的特征
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=bs)

    # reshape特征，使得每一个图像都是展平的‘MaxPooling2D‘输出的特征向量
    features = features.reshape((features.shape[0], 2048 * 7 * 7))

    # 特征和标签添加进HDF5数据集
    dataset.add(features, batchLabels)
    pbar.update(i)

# 关闭数据集
dataset.close()
pbar.finish()

