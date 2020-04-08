import time
import logging
import os
from tqdm import tqdm

from utils import unet_dataset
from models import unet, unetPlusPlus, attentionUnet
from config import config
from metrics import eval_metrics

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

logger = logging.getLogger()
logger.setLevel(logging.INFO)

rq = time.strftime('%Y%m%d%H%M', time.strftime(time.time()))
log_path = r'..\logs'
log_name = os.path.join(log_path, rq + '.log')
logfile = log_name
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)

def train():

    model = attentionUnet.AttentionUnet(num_classes=config.num_classes)
    model.cuda()
    # loss
    criterion = nn.CrossEntropyLoss().cuda()

    # train data
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
    )
    dst_train = unet_dataset.UnetDataset(r'.\data\GID_5classes\train_log.txt', transform=transform)
    dataloader_train = DataLoader(dst_train, shuffle=True, batch_size=config.batch_size)

    # validation data
    transform = transforms.Compose(
        [
            #transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
    )
    dst_valid = unet_dataset.UnetDataset(r'.\data\GID_5classes\val_log.txt', transform=transform)
    dataloader_valid = DataLoader(dst_valid, shuffle=False, batch_size=config.batch_size)


    for epoch in range(config.num_epoch):

        epoch_start = time.time()

        # lr
        lr = 0.0001

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0002)


        model.train()

        loss_sum = 0.0
        correct_sum = 0.0
        labeled_sum = 0.0
        inter_sum = 0.0
        unoin_sum = 0.0
        tbar = tqdm(dataloader_train, ncols=120)
        for batch_idx, (data, target) in enumerate(tbar):
            tic = time.time()

            #data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()

            correct, labeled, inter, unoin = eval_metrics(output, target, config.num_classes)
            correct_sum += correct
            labeled_sum += labeled
            inter_sum += inter
            unoin_sum += unoin
            pixelAcc = 1.0 * correct_sum / (np.spacing(1)+labeled_sum)
            mIoU = 1.0 * inter_sum / (np.spacing(1) + unoin_sum)
            tbar.set_description('TRAIN ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.2f} | bt {:.2f} et {:.2f}|'.format(
                epoch, loss_sum/((batch_idx+1)*config.batch_size),
                pixelAcc, mIoU.mean(),
                time.time()-tic, time.time()-epoch_start))
            logger.info('TRAIN ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.2f} | bt {:.2f} et {:.2f}|'.format(
                epoch, loss_sum/((batch_idx+1)*config.batch_size),
                pixelAcc, mIoU.mean(),
                time.time()-tic, time.time()-epoch_start))
        if epoch % 5 == 0:
            max_pixACC = 0.0
            model.eval()
            loss_sum = 0.0
            correct_sum = 0.0
            labeled_sum = 0.0
            inter_sum = 0.0
            unoin_sum = 0.0
            pixelAcc = 0.0
            tbar = tqdm(dataloader_valid, ncols=120)
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(tbar):
                    tic = time.time()

                    # data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    loss_sum += loss.item()

                    correct, labeled, inter, unoin = eval_metrics(output, target, config.num_classes)
                    correct_sum += correct
                    labeled_sum += labeled
                    inter_sum += inter
                    unoin_sum += unoin
                    pixelAcc = 1.0 * correct_sum / (np.spacing(1) + labeled_sum)
                    mIoU = 1.0 * inter_sum / (np.spacing(1) + unoin_sum)
                    tbar.set_description('VAL ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.2f} | bt {:.2f} et {:.2f}|'.format(
                        epoch, loss_sum / ((batch_idx + 1) * config.batch_size),
                        pixelAcc, mIoU.mean(),
                               time.time() - tic, time.time() - epoch_start))
                    logger.info('VAL ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.2f} | bt {:.2f} et {:.2f}|'.format(
                        epoch, loss_sum / ((batch_idx + 1) * config.batch_size),
                        pixelAcc, mIoU.mean(),
                               time.time() - tic, time.time() - epoch_start))
                if pixelAcc > max_pixACC:
                    max_pixACC = pixelAcc
                    torch.save(model.state_dict(), r".\saved\unetpp.pth")
if __name__ == '__main__':
    train()