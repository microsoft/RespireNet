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

# load external modules
from utils import *
from image_dataloader import *
from nets.network_cnn import *
#from nets.network_hybrid import *
from sklearn.metrics import confusion_matrix, accuracy_score
print ("Train import done successfully")

# input argmuments
parser = argparse.ArgumentParser(description='RespireNet: Lung Sound Classification')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=0.0005,help='weight decay value')
parser.add_argument('--gpu_ids', default=[0,1], help='a list of gpus')
parser.add_argument('--num_worker', default=4, type=int, help='numbers of worker')
parser.add_argument('--batch_size', default=4, type=int, help='bacth size')
parser.add_argument('--epochs', default=10, type=int, help='epochs')
parser.add_argument('--start_epochs', default=0, type=int, help='start epochs')

parser.add_argument('--data_dir', type=str, help='data directory')
parser.add_argument('--folds_file', type=str, help='folds text file')
parser.add_argument('--test_fold', default=4, type=int, help='Test Fold ID')
parser.add_argument('--stetho_id', default=-1, type=int, help='Stethoscope device id')
parser.add_argument('--aug_scale', default=None, type=float, help='Augmentation multiplier')
parser.add_argument('--model_path',type=str, help='model saving directory')
parser.add_argument('--checkpoint', default=None, type=str, help='load checkpoint')

args = parser.parse_args()

################################MIXUP#####################################
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

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
        mean, std = get_mean_and_std(image_loader(self.args.data_dir, self.args.folds_file, 
            self.args.test_fold, True, "Params_json", Compose([ToTensor()]), stetho_id=self.args.stetho_id))
        print("MEAN",  mean, "STD", std)

        self.input_transform = Compose([ToTensor(), Normalize(mean, std)])
        train_dataset = image_loader(self.args.data_dir, self.args.folds_file, self.args.test_fold, 
                True, "params_json", self.input_transform, stetho_id=self.args.stetho_id, aug_scale=self.args.aug_scale)
        test_dataset = image_loader(self.args.data_dir, self.args.folds_file, self.args.test_fold, 
                False, "params_json", self.input_transform, stetho_id=self.args.stetho_id)
        self.test_ids = np.array(test_dataset.identifiers)
        self.test_paths = test_dataset.filenames_with_labels

        # loading checkpoint
        self.net = model(num_classes=4).cuda()
        if self.args.checkpoint is not None:
            checkpoint = torch.load(self.args.checkpoint)
            self.net.load_state_dict(checkpoint)
            # uncomment in case fine-tuning, specify block layer
            # before block_layer, all layers will be frozen durin training
            #self.net.fine_tune(block_layer=5)
            print("Pre-trained Model Loaded:", self.args.checkpoint)
        self.net = nn.DataParallel(self.net, device_ids=self.args.gpu_ids)

        # weighted sampler
        reciprocal_weights = []
        for idx in range(len(train_dataset)):
            reciprocal_weights.append(train_dataset.class_probs[train_dataset.labels[idx]])
        weights = (1 / torch.Tensor(reciprocal_weights))
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_dataset))

        self.train_data_loader = DataLoader(train_dataset, num_workers=self.args.num_worker, 
                batch_size=self.args.batch_size, sampler=sampler)
        self.val_data_loader = DataLoader(test_dataset, num_workers=self.args.num_worker, 
                batch_size=self.args.batch_size, shuffle=False)
        print("DATA LOADED")

        print("Params to learn:")
        params_to_update = []
        for name,param in self.net.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

        # Observe that all parameters are being optimized
        self.optimizer = optim.SGD(params_to_update, lr=self.args.lr, momentum=0.9, weight_decay=self.args.weight_decay)
        #self.optimizer = optim.Adam(params_to_update, lr=self.args.lr, weight_decay=self.args.weight_decay)

        # Decay LR by a factor
        #self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.33)

        # weights for the loss function
        weights = torch.tensor([3.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        #weights = torch.tensor(train_dataset.class_probs, dtype=torch.float32)
        weights = weights / weights.sum()
        weights = 1.0 / weights
        weights = weights / weights.sum()
        weights = weights.cuda()
        self.loss_func = nn.CrossEntropyLoss(weight=weights)
        self.loss_nored = nn.CrossEntropyLoss(reduction='none')

    def evaluate(self, net, epoch, iteration):

        self.net.eval()
        test_losses = []
        class_hits = [0.0, 0.0, 0.0, 0.0] # normal, crackle, wheeze, both
        class_counts = [0.0, 0.0, 0.0+1e-7, 0.0+1e-7] # normal, crackle, wheeze, both
        running_corrects = 0.0
        denom = 0.0

        classwise_test_losses = [[], [], [], []]
        conf_label, conf_pred = [], []
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
        #print("Classwise_Losses Normal: {}, Crackle: {}, Wheeze: {}, Both: {}".format(np.mean(classwise_test_losses[0]),
        #    np.mean(classwise_test_losses[1]), np.mean(classwise_test_losses[2]), np.mean(classwise_test_losses[3])))
        #get_score(class_hits, class_counts, True)

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
        self.net.train()

        return acc, np.mean(test_losses)

    def train(self):
        train_losses = []
        test_losses = []
        test_acc = []
        best_acc = -1

        for _, epoch in tqdm(enumerate(range(self.args.start_epochs, self.args.epochs))):
            losses = []
            class_hits = [0.0, 0.0, 0.0, 0.0]
            class_counts = [0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7]
            running_corrects = 0.0
            denom = 0.0
            classwise_train_losses = [[], [], [], []]
                
            for i, (image, label) in tqdm(enumerate(self.train_data_loader)):
                
                image, label = image.cuda(), label.cuda()
                # in case using mixup, uncomment 2 lines below
                #image, label_a, label_b, lam = mixup_data(image, label, alpha=0.5)
                #image, label_a, label_b = map(Variable, (image, label_a, label_b))

                output = self.net(image)

                # calculate loss from output
                # in case using mixup, uncomment line below and comment the next line
                #loss = mixup_criterion(self.loss_func, output, label_a, label_b, lam)
                loss = self.loss_func(output, label)
                loss_nored = self.loss_nored(output, label)

                _, preds = torch.max(output, 1)
                running_corrects += torch.sum(preds == label.data)
                denom += len(label.data)

                #class
                for idx in range(preds.shape[0]):
                    class_counts[label[idx].item()] += 1.0
                    if preds[idx].item() == label[idx].item():
                         class_hits[label[idx].item()] += 1.0
                    classwise_train_losses[label[idx].item()].append(loss_nored[idx].item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.data.cpu().numpy())

                if i % 1000 == self.train_data_loader.__len__()-1:
                    print("---------------------------------------------")
                    print("epoch {} iter {}/{} Train Total loss: {}".format(epoch,
                        i, len(self.train_data_loader), np.mean(losses)))
                    print("Train Accuracy: {}".format(running_corrects.double() / denom))
                    print("Classwise_Losses Normal: {}, Crackle: {}, Wheeze: {}, Both: {}".format(np.mean(classwise_train_losses[0]),
                        np.mean(classwise_train_losses[1]), np.mean(classwise_train_losses[2]), np.mean(classwise_train_losses[3])))
                    get_score(class_hits, class_counts, True)

                    print("testing......")
                    acc, test_loss = self.evaluate(self.net, epoch, i)

                    if best_acc < acc:
                        best_acc = acc
                        torch.save(self.net.module.state_dict(), args.model_path+'/ckpt_best_'+str(self.args.epochs)+'_'+str(self.args.stetho_id)+'.pkl')
                        print("Best ACC achieved......", best_acc.item())
                    print("BEST ACCURACY TILL NOW", best_acc)

                    train_losses.append(np.mean(losses))
                    test_losses.append(test_loss)
                    test_acc.append(acc)
            #self.exp_lr_scheduler.step()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
