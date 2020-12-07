#!/usr/bin/python                                                       
# Author: Siddhartha Gairola (t-sigai at microsoft dot com))                 
                                                                    
import os
import itertools
import argparse
import random
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler


import torchvision
from torchvision.transforms import Compose, Normalize, ToTensor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# load external modules
from utils import *
from image_dataloader import *
from nets.network_cnn import *

print ("Train import done successfully")

# input argmuments
parser = argparse.ArgumentParser(description='Lung Sound Classification')
parser.add_argument('--steth_id', default=-1.0, type=float, help='learning rate')
parser.add_argument('--gpu_ids', default=[0,1], help='a list of gpus')
parser.add_argument('--num_worker', default=4, type=int, help='numbers of worker')
parser.add_argument('--batch_size', default=4, type=int, help='bacth size')
parser.add_argument('--data_dir', type=str, help='data directory')
parser.add_argument('--folds_file', type=str, help='folds text file')
parser.add_argument('--test_fold', default=4, type=int, help='Test Fold ID')
parser.add_argument('--checkpoint', default=None, type=str, help='load checkpoint')

args = parser.parse_args()

##############################################################################
def get_score(hits, counts, pflag=False):
    se = (hits[1] + hits[2] + hits[3]) / (counts[1] + counts[2] + counts[3])
    sp = hits[0] / counts[0]
    sc = (se+sp) / 2.0

    if pflag:
        print("*************Metrics******************")
        print("Se: {}, Sp: {}, Score: {}".format(se, sp, sc))
        print("Normal: {}, Crackle: {}, Wheeze: {}, Both: {}".format(hits[0]/counts[0], hits[1]/counts[1], 
            hits[2]/counts[2], hits[3]/counts[3]))
    
class Trainer:
    def __init__(self):
        self.args = args

        mean, std = [0.5091, 0.1739, 0.4363], [0.2288, 0.1285, 0.0743]
        self.input_transform = Compose([ToTensor(), Normalize(mean, std)])

        test_dataset = image_loader(self.args.data_dir, self.args.folds_file, self.args.test_fold, 
                False, "params_json", self.input_transform, self.args.steth_id)
        self.test_ids = np.array(test_dataset.identifiers)
        self.test_paths = test_dataset.filenames_with_labels

        # loading checkpoint
        self.net = model(num_classes=4).cuda()
        if self.args.checkpoint is not None:
            checkpoint = torch.load(self.args.checkpoint)
            self.net.load_state_dict(checkpoint)
            self.net.fine_tune(block_layer=5)
            print("Pre-trained Model Loaded:", self.args.checkpoint)
        self.net = nn.DataParallel(self.net, device_ids=self.args.gpu_ids)

        self.val_data_loader = DataLoader(test_dataset, num_workers=self.args.num_worker, 
                batch_size=self.args.batch_size, shuffle=False)
        print("Test Size", len(test_dataset))
        print("DATA LOADED")

        self.loss_func = nn.CrossEntropyLoss()
        self.loss_nored = nn.CrossEntropyLoss(reduction='none')

    def evaluate(self, net, epoch, iteration):

        self.net.eval()
        test_losses = []
        class_hits = [0.0, 0.0, 0.0, 0.0] # normal, crackle, wheeze, both
        class_counts = [0.0, 0.0, 0.0+1e-7, 0.0+1e-7] # normal, crackle, wheeze, both
        running_corrects = 0.0
        denom = 0.0

        classwise_test_losses = [[], [], [], []]

        conf_label = []
        conf_pred = []
        for i, (image, label) in tqdm(enumerate(self.val_data_loader)):
            image, label = image.cuda(), label.cuda()
            output = self.net(image)
            
            # calculate loss from output
            loss = self.loss_func(output, label)
            loss_nored = self.loss_nored(output, label)
            test_losses.append(loss.data.cpu().numpy())
            
            _, preds = torch.max(output, 1)
            running_corrects += torch.sum(preds == label.data)

            # updating denom
            denom += len(label.data)

            #class
            for idx in range(preds.shape[0]):
                class_counts[label[idx].item()] += 1.0
                conf_label.append(label[idx].item())
                conf_pred.append(preds[idx].item())
                if preds[idx].item() == label[idx].item():
                    class_hits[label[idx].item()] += 1.0

                classwise_test_losses[label[idx].item()].append(loss_nored[idx].item())

        print("Val Accuracy: {}".format(running_corrects.double() / denom))
        print("epoch {}, Validation BCE loss: {}".format(epoch, np.mean(test_losses)))

        #aggregating same id, majority voting
        conf_label = np.array(conf_label)
        conf_pred = np.array(conf_pred)
        y_pred, y_true = [], []
        for pt in self.test_paths:
            y_pred.append(np.argmax(np.bincount(conf_pred[np.where(self.test_ids == pt)])))
            y_true.append(int(pt.split('_')[-1]))

        conf_matrix = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        print("Confusion Matrix", conf_matrix)
        print("Accuracy Score", acc)

        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:,np.newaxis]
        print("Classwise Scores", conf_matrix.diagonal())
        return acc, np.mean(test_losses)

if __name__ == "__main__":
    trainer = Trainer()
    acc, test_loss = trainer.evaluate(trainer.net, 0, 0)
