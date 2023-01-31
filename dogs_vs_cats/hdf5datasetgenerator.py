from keras.utils import np_utils
import numpy as np
import h5py

class HDF5DatasetGenerator:
    def __init__(self, dbPath, batchSize, preprocessors=None,
        aug=None, binarize=True, classes=2):
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes

        self.db = h5py.File(dbPath)
        self.numImages = self.db["labels"].shape[0]

    def generator(self, passes=np.inf):
        # 初始化epoch
        epochs = 0

        # 一直循环，直到到达设置的epochs
        while epochs < passes:
            # 循环HDF5数据集
            for i in np.arange(0, self.numImages, self.batchSize):
                # 提取图像和标签
                images = self.db["images"][i: i + self.batchSize]
                labels = self.db["labels"][i: i + self.batchSize]

                # 看下标签是否二值化
                if self.binarize:
                    labels = np_utils.to_categorical(labels,
                                                     self.classes)

                # 确定是否使用预处理器
                if self.preprocessors is not None:
                    procImages = []
                    
                    # 循环图像
                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)
                        
                        procImages.append(image)
                    
                    images = np.array(procImages)

                # 应用数据增强
                if self.aug is not None:
                    (images, labels) = next(self.aug.flow(images,
                                                          labels, batch_size=self.batchSize))

                # 以元组格式返回图像和标签
                yield (images, labels)
                
            # 递增epochs
            epochs += 1
            
    def close(self):
        # 关闭数据集
        self.db.close()
