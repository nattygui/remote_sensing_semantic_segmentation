import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from models.unet import UNet

img_path = r"I:\dataset\GID\Large-scale Classification_5classes\temp"


def predict():
    img_path = r"I:\dataset\GID\Large-scale Classification_5classes\temp"

    im_w, im_h = 256, 256
    crop_w, crop_h = 2, 2
    stride_w, stride_h = im_w // crop_w, im_h // crop_h
    check_point = r".\saved\unet.pth"

    classes = [0, 1, 2, 3, 4]

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
    )

    model = UNet(len(classes))
    model.load_state_dict(torch.load(check_point))
    model.cuda()
    model.eval()

    for im in os.listdir(img_path):
        image_path = os.path.join(img_path, im)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, _ = image.shape
        padding_h = (h//im_h + 1) * im_h
        padding_w = (w//im_w + 1) * im_w
        padding_image = np.zeros((padding_h, padding_w, 3), dtype=np.uint8) + 4
        padding_image[0:h, 0:w, :] = image[:, :, :]
        mask_image = np.zeros((padding_image.shape[0], padding_image.shape[1]), dtype=np.uint8) + 4
        padding_image = np.asarray(padding_image, dtype=np.uint8)

        with torch.no_grad():
            for i in tqdm(range(padding_h//im_h)):
                for j in range(padding_w//im_w):
                    crop_image = padding_image[i*im_h:(i+1)*im_h,j*im_w:(j+1)*im_w, :]
                    crop_image = transform(crop_image).cuda()
                    crop_image = crop_image.reshape(1, 3, im_h, im_w)

                    output = model(crop_image)

                    _, pred = output.max(1)
                    pred = pred.view(256, 256)

                    start_h, start_w = i*im_h, j*im_w
                    for si in range(stride_h):
                        for sj in range(stride_w):
                            #temp =  pred.cpu().numpy()[si*crop_h : (si+1)*crop_h, sj*crop_w : (sj+1)*crop_w]
                            mask_image[start_h+si*crop_h : start_h+(si+1)*crop_h,
                            start_w+sj*crop_w : start_w+(sj+1)*crop_w] = \
                                pred.cpu().numpy()[si*crop_h : (si+1)*crop_h, sj*crop_w : (sj+1)*crop_w]
            save_label = os.path.join(r".\data\GID_15classes\predict_label", im)
            # cv2.imshow('show', mask_image[0:h, 0:w])
            # cv2.waitKey(0)
            cv2.imwrite(save_label, mask_image[0:h, 0:w])
            save_visual = os.path.join(r".\data\GID_15classes\predict_visual", im.split('.')[0] + 'visual.tif')
            translabeltovisual(save_label, save_visual)


def translabeltovisual(save_label, path):
    num_classes5 = [[255, 0, 0], [0, 255, 0], [0, 255, 255], [255, 255, 0], [0, 0, 255]]
    num_classes = [
        [200, 0, 0],
        [250, 0, 150],
        [200, 150, 150],
        [250, 150, 150],

        [0, 200, 0],
        [150, 250, 0],
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
    im = cv2.imread(save_label)
    for i in tqdm(range(im.shape[0])):
        for j in range(im.shape[1]):
            rgb = [im[i][j][0], im[i][j][1], im[i][j][2]]
            if rgb == [0, 0, 0]:
                im[i][j] = num_classes5[0]
            elif rgb == [1, 1, 1]:
                im[i][j] = num_classes5[1]
            elif rgb == [2, 2, 2]:
                im[i][j] = num_classes5[2]
            elif rgb == [3, 3, 3]:
                im[i][j] = num_classes5[3]
            elif rgb == [4, 4, 4]:
                im[i][j] = num_classes5[4]

    cv2.imwrite(path, im)
if __name__ == '__main__':
    # im = "GF2_PMS1__L1A0000564539-MSS1.tif"
    # save_label = os.path.join(r"I:\learn\remote_sensing_semantic_segmentation\data\GID_5classes\predict_label", im)
    # save_visual = os.path.join(r"I:\learn\remote_sensing_semantic_segmentation\data\GID_5classes\predict_label",
    #                            im.split('.')[0] + 'visual.tif')
    # translabeltovisual(save_label, save_visual)
    predict()
