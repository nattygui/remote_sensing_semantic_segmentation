import os
import random
def createImageLabelList():
    origin_path = r'I:\dataset\remote_sensing_image\remote_sensing_image\train'
    train_log = open(os.path.join(origin_path, 'train_log.txt'), 'w')
    val_log = open(os.path.join(origin_path,'val_log.txt'), 'w')

    imagepath = os.path.join(origin_path, 'src')
    labelpath = os.path.join(origin_path, 'label')

    images = os.listdir(imagepath)
    random.shuffle(images)

    for i in images[:int(len(images)*0.75)]:
        train_log.write(os.path.abspath(os.path.join(imagepath, i)) + " " + os.path.abspath(os.path.join(labelpath, i)) + "\n")

    for i in images[int(len(images)*0.75):]:
        val_log.write(os.path.abspath(os.path.join(imagepath, i)) + " " + os.path.abspath(os.path.join(labelpath, i)) + "\n")

if __name__ == "__main__":
    createImageLabelList()
