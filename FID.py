from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
import tensorflow as tf
import numpy as np
from numpy import asarray
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm

def fid_scale_images(images, new_shape):
    images_list = list()
    for image in images:
        new_image = resize(image, new_shape, 0)
        images_list.append(new_image)
    return asarray(images_list)


def calculate_fid(fid_model, images1, images2):
    images1 = images1.astype('float32')
    images2 = images2.astype('float32')
    images1 = fid_scale_images(images1, (299, 299, 3))
    images2 = fid_scale_images(images2, (299, 299, 3))
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)
    act1 = fid_model.predict(images1)
    act2 = fid_model.predict(images2)
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if iscomplexobj(covmean):
        covmean = covmean.real
    fid_score = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid_score

def init_fid():
    inc3 = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    return inc3