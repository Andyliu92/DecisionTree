import os
from skimage import data
from skimage import io
import numpy as np
from skimage import transform
from skimage.color import rgb2gray
 
def loadData(data_directory):
    directories = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) for f in os.listdir(label_directory) if f.endswith(".ppm")]
        for f in file_names:
            images.append(io.imread(f))
            labels.append(int(d))
    images = __transformRgb2gray(__transformResize(images))
    images = images.reshape((images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.array(labels)
    print("Finish loading dataset")
    return images, labels
 
def __transformResize(images):
    images16 = [transform.resize(image, (16, 16)) for image in images]
    images16 = np.array(images16)
    return images16
 
def __transformRgb2gray(images):
    images = rgb2gray(images)
    return images


if __name__ == "__main__":
    images, labels = loadData("/home/ubuntu/research/FuncCAM/data/BelgiumTSC/Training")
