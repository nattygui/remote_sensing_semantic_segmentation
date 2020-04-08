import cv2
import random
import os
import numpy as np
from tqdm import tqdm

img_w = 256
img_h = 256
imagePath = r"I:\dataset\GID\Large-scale Classification_5classes\image_RGB"
labelPath = r"I:\dataset\GID\Large-scale Classification_5classes\label_5classes"

def getImagesAndLabels(imagePath, labelPath):
    images, labels = [], []
    for im, label in zip(os.listdir(imagePath),os.listdir(labelPath)):
        if im.split('.')[-1] != 'tif' and label.split('.')[-1] != 'tif':
            continue
        im = os.path.join(imagePath, im)
        label = os.path.join(labelPath, label)
        images.append(im)
        labels.append(label)
    return images, labels


def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb, yb


def blur(img):
    img = cv2.blur(img, (3, 3))
    return img


def add_noise(img):
    for i in range(200):  # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img


def data_augment(xb, yb):
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 90)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 180)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)

    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb, 1.0)

    if np.random.random() < 0.25:
        xb = blur(xb)

    if np.random.random() < 0.2:
        xb = add_noise(xb)

    return xb, yb

def tranferRGBClasses5(label):
    """
    5分类
    build-up 255 0 0 RED
    farmland 0 255 0 Green1
    forest 0 255 255 Cyan1 淡蓝
    meadow 255 255 0 Yellow1
    water 0 0 255 Blue
    other 0 0 0
    """
    back = [5, 5, 5]
    for i in tqdm(range(label.shape[0])):
        for j in range(label.shape[1]):
            # 清除噪声点
            if 0<label[i][j][0]<255 : label[i][j][0]=255
            if 0 < label[i][j][1] < 255: label[i][j][1] = 255
            if 0 < label[i][j][2] < 255: label[i][j][2] = 255

            if label[i][j][0] == 255 and label[i][j][1] == 0 and label[i][j][2] == 0:
                label[i][j] = [0, 0, 0]
            elif label[i][j][0] == 0 and label[i][j][1] == 255 and label[i][j][2] == 0:
                label[i][j] = [1, 1, 1]
            elif label[i][j][0] == 0 and label[i][j][1] == 255 and label[i][j][2] == 255:
                label[i][j] = [2, 2, 2]
            elif label[i][j][0] == 255 and label[i][j][1] == 255 and label[i][j][2] == 0:
                label[i][j] = [3, 3, 3]
            elif label[i][j][0] == 0 and label[i][j][1] == 0 and label[i][j][2] == 255:
                label[i][j] = [4, 4, 4]
            elif label[i][j][0] == 0 and label[i][j][1] == 0 and label[i][j][2] == 0:
                label[i][j] = [5, 5, 5]
            else:
                label[i][j] = back
            back = label[i][j]
    return label

def tranferLabels():
    """
        将所有彩色标签图像转化成对应标签值图像
        label information of 5 classes:
        built-up  RGB:    255,    0,    0    label:0
        farmland  RGB:    0,    255,    0    label:1
        forest    RGB:  0,    255,  255      label:2
        meadow    RGB:  255, 255,     0      label:3
        water     RGB:  0,        0,  255    label:4
        ##############################
        label information of 15 classes:
        industrial land RGB:    200,    0,    0   label:0
        urban residential RGB:    250,    0, 150    label:1
        rural residential RGB:    200, 150, 150     label:2
        traffic land RGB:    250, 150, 150          label:3

        paddy field RGB:    0,     200,    0        label:4
        irrigated land RGB:    150,  250,   0       label:5
        dry cropland RGB:    150, 200, 150          label:6

        garden plot RGB:    200,     0, 200         label:7
        arbor woodland RGB:    150,     0, 250      label:8
        shrub land RGB:    150,  150, 250           label:9

        natural grassland RGB:    250,  200,    0   label:10
        artificial grassland RGB:    200,  200,  0  label:11

        river RGB:    0,         0, 200             label:12
        lake RGB:    0,     150, 200                label:13
        pond RGB:    0,     200, 250                label:14
    """
    num_classes5 = [[255, 0, 0], [0, 255, 0], [0, 255, 255], [255, 255, 0], [0, 0, 255]]
    pre_labels_path = r"I:\dataset\GID\Large-scale Classification_5classes\label_5classes"
    post_labels_path = r"I:\dataset\GID\Large-scale Classification_5classes\label_label"
    for label_path in os.listdir(pre_labels_path):
        label = cv2.imread(os.path.join(pre_labels_path, label_path))
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        label = tranferRGBClasses5(label)
        post_label_path = os.path.join(post_labels_path, label_path)
        cv2.imwrite(post_label_path, label)
def ChannelTo1Channel():
    labels_path = r"I:\learn\remote_sensing_semantic_segmentation\data\GID_5classes\train\label"
    for label in os.listdir(labels_path):
        im = cv2.imread(os.path.join(labels_path, label), cv2.IMREAD_GRAYSCALE)
        label_path = os.path.join(labels_path, label)
        cv2.imwrite(label_path, im)
def trans15CImageTo5Label():
    num_classes = [
        [[0, 200, 0], [150, 250, 0], [150, 250, 0]], # 农田 label 0
        [[250, 200, 0], [200, 200, 0]], # 草地 label 1
        [[200, 0, 200], [150, 150, 250]], # 灌木 label 2
        [[150, 0, 250]] # 林地 label 3  #other 4
    ]
    labels_path = r"I:\dataset\GID\Fine Land-cover Classification_15classes\image_RGB"
    for label in tqdm(os.listdir(labels_path)):
        image = cv2.imread(os.path.join(labels_path, label))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                temp_rgb = [image[i][j][0], image[i][j][1], image[i][j][2]]

                if temp_rgb in num_classes[0]:
                    image[i][j] = [0, 0, 0]
                elif temp_rgb in num_classes[1]:
                    image[i][j] = [1, 1, 1]
                elif temp_rgb in num_classes[2]:
                    image[i][j] = [2, 2, 2]
                elif temp_rgb in num_classes[3]:
                    image[i][j] = [3, 3, 3]
                else:
                    image[i][j] = [4, 4, 4]
        post_label_path = os.path.join(r'I:\learn\remote_sensing_semantic_segmentation\data\GID_15classes\large_label', label)
        cv2.imwrite(post_label_path, image)

def trans15CImageTo15Label():
    num_classes = [
        [200, 0, 0],
        [250, 0, 150],
        [200, 150, 150],
        [250, 150, 150],

        [0, 200, 0],
        [150,  250,   0],
        [150, 200, 150],

        [200, 0, 200],
        [150, 0, 250],
        [150, 150, 250],

        [250, 200, 0],
        [200, 200, 0],

        [0, 0, 200],
        [0, 150, 200],
        [0, 200, 250]
    ]
    labels_path = r"I:\dataset\GID\Fine Land-cover Classification_15classes\label_15classes"
    for label in os.listdir(labels_path):
        image = cv2.imread(os.path.join(labels_path, label))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for i in tqdm(range(image.shape[0])):
            for j in range(image.shape[1]):
                temp_rgb = [image[i][j][0], image[i][j][1], image[i][j][2]]
                for num in range(len(num_classes)+1):
                    if num == len(num_classes):
                        image[i][j] = [num, num, num]
                        break
                    if temp_rgb == num_classes[num]:
                        image[i][j] = [num, num, num]
                        break


        post_label_path = os.path.join(r'I:\dataset\GID\Fine Land-cover Classification_15classes\label_label',
                                       label)
        cv2.imwrite(post_label_path, image)
def creat_dataset(image_num=10000, mode='original'):
    print('creating dataset...')
    image_sets, label_sets = getImagesAndLabels(imagePath, labelPath)
    image_each = image_num / len(image_sets)
    g_count = 0
    for i in tqdm(range(len(image_sets))):
        count = 0
        src_img = cv2.imread(image_sets[i])  # 3 channels
        label_img = cv2.imread(label_sets[i], cv2.IMREAD_GRAYSCALE)
        X_height, X_width, _ = src_img.shape
        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            try:
                src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w, :]
                label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]
            except:
                print("{}当前文件错误".format(image_sets[i]))
                break
            if mode == 'augment':
                src_roi, label_roi = data_augment(src_roi, label_roi)

            # visualize = np.zeros((256, 256)).astype(np.uint8)
            # visualize = label_roi * 50

            #cv2.imwrite(('./unet_train/visualize/%d.png' % g_count), visualize)
            cv2.imwrite((r'..\data\GID_15classes\image\%d.png' % g_count), src_roi)
            cv2.imwrite((r'..\data\GID_15classes\label\%d.png' % g_count), label_roi)
            count += 1
            g_count += 1


if __name__ == '__main__':
    # creat_dataset(mode='augment')
    tranferLabels()
    # ChannelTo1Channel()
    #  trans15CImageTo15Label()