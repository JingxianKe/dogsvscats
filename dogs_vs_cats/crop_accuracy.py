import dogs_vs_cats_config as config
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from simplepreprocessor import SimplePreprocessor
from meanpreprocessor import MeanPreprocessor
from croppreprocessor import CropPreprocessor
from hdf5datasetgenerator import HDF5DatasetGenerator
from ranked import rank5_accuracy
from keras.models import load_model
import numpy as np
import progressbar
import json

# 加载训练集的RGB均值
means = json.loads(open(config.DATASET_MEAN).read())

# 初始化图像预处理
sp = SimplePreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
cp = CropPreprocessor(227, 227)
iap = ImageToArrayPreprocessor()

# 加载预训练网络
print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)

# 初始化测试集生成器，进行预测
print("[INFO] predicting on test data (no crops)...")
testGen = HDF5DatasetGenerator(config.TEST_HDF5, 64,
							   preprocessors=[sp, mp, iap], classes=2)
predictions = model.predict_generator(testGen.generator(),
									  steps=testGen.numImages // 64, max_queue_size=64 * 2)

# 计算rank-1和rank-5准确率
(rank1, _) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
testGen.close()

# 重新初始化测试集生成器，这次不包括‘SimplePreprocessor‘
testGen = HDF5DatasetGenerator(config.TEST_HDF5, 64,
							   preprocessors=[mp], classes=2)
predictions = []

# 显示进度条
widgets = ["Evaluating: ", progressbar.Percentage(), " ",
		   progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=testGen.numImages // 64,
							   widgets=widgets).start()

for (i, (images, labels)) in enumerate(testGen.generator(passes=1)):
	for image in images:
		# 使用crop preprocessor生成10个separate crops，之后把图像转换为数组
		crops = cp.preprocess(image)
		crops = np.array([iap.preprocess(c) for c in crops],
						 dtype="float32")
		
		# 预测
		pred = model.predict(crops)
		predictions.append(pred.mean(axis=0))
		
	# 进度条更新
	pbar.update(i)

# 计算rank-1准确率
pbar.finish()
print("[INFO] predicting on test data (with crops)...")
(rank1, _) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
testGen.close()
