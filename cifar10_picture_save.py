from keras.datasets import cifar10
from multiprocessing import Pool, cpu_count
import cv2
import numpy as np
from tqdm import tqdm

def save_image(n):
    array = x_test[n]
    # array = array.transpose(1,2,0)
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return cv2.imwrite("cifar10/image" + str(n) + ".png", array)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

pool = Pool(cpu_count())
images = list(tqdm(pool.imap(save_image, range(len(x_test))), total=len(x_test)))
pool.close()
pool.join()