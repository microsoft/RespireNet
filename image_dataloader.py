#!/usr/bin/python                                                       
# Author: Siddhartha Gairola (t-sigai at microsoft dot com)

import os
import random
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

import librosa
from tqdm import tqdm

from utils import *

class image_loader(Dataset):
    def __init__(self, data_dir, folds_file, test_fold, train_flag, params_json, input_transform=None, stetho_id=-1, aug_scale=None):

        # getting device-wise information
        self.file_to_device = {}
        device_to_id = {}
        device_id = 0
        files = os.listdir(data_dir)
        device_patient_list = []
        pats = []
        for f in files:
            device = f.strip().split('_')[-1].split('.')[0]
            if device not in device_to_id:
                device_to_id[device] = device_id
                device_id += 1
                device_patient_list.append([])
            self.file_to_device[f.strip().split('.')[0]] = device_to_id[device]
            pat = f.strip().split('_')[0]
            if pat not in device_patient_list[device_to_id[device]]:
                device_patient_list[device_to_id[device]].append(pat)
            if pat not in pats:
                pats.append(pat)

        print("DEVICE DICT", device_to_id)
        for idx in range(device_id):
            print("Device", idx, len(device_patient_list[idx]))

        # get patients dict in current fold based on train flag
        all_patients = open(folds_file).read().splitlines()
        patient_dict = {}
        for line in all_patients:
            idx, fold = line.strip().split(' ')
            if train_flag and int(fold) != test_fold:
                patient_dict[idx] = fold
            elif train_flag == False and int(fold) == test_fold:
                patient_dict[idx] = fold

        #extracting the audiofilenames and the data for breathing cycle and it's label
        print("Getting filenames ...")
        filenames, rec_annotations_dict = get_annotations(data_dir)
        if stetho_id >= 0:
            self.filenames = [s for s in filenames if s.split('_')[0] in patient_dict and self.file_to_device[s] == stetho_id]
        else:
            self.filenames = [s for s in filenames if s.split('_')[0] in patient_dict]

        self.audio_data = [] # each sample is a tuple with id_0: audio_data, id_1: label, id_2: file_name, id_3: cycle id, id_4: aug id, id_5: split id
        self.labels = []
        self.train_flag = train_flag
        self.data_dir = data_dir
        self.input_transform = input_transform

        # parameters for spectrograms
        self.sample_rate = 4000
        self.desired_length = 8
        self.n_mels = 64
        self.nfft = 256
        self.hop = self.nfft//2
        self.f_max = 2000

        self.dump_images = False
        self.filenames_with_labels = []
        
        # get individual breathing cycles from each audio file
        print("Exracting Individual Cycles")
        self.cycle_list = []
        self.classwise_cycle_list = [[], [], [], []]
        for idx, file_name in tqdm(enumerate(self.filenames)):
            data = get_sound_samples(rec_annotations_dict[file_name], file_name, data_dir, self.sample_rate)
            cycles_with_labels = [(d[0], d[3], file_name, cycle_idx, 0) for cycle_idx, d in enumerate(data[1:])]
            self.cycle_list.extend(cycles_with_labels)
            for cycle_idx, d in enumerate(cycles_with_labels):
                self.filenames_with_labels.append(file_name+'_'+str(d[3])+'_'+str(d[1]))
                self.classwise_cycle_list[d[1]].append(d)
        
        # concatenation based augmentation scheme
        if train_flag and aug_scale:
            self.new_augment(scale=aug_scale)

        # split and pad each cycle to the desired length
        for idx, sample in enumerate(self.cycle_list):
            output = split_and_pad(sample, self.desired_length, self.sample_rate, types=1)
            self.audio_data.extend(output)

        self.device_wise = []
        for idx in range(device_id):
            self.device_wise.append([])
        self.class_probs = np.zeros(4)
        self.identifiers = []
        for idx, sample in enumerate(self.audio_data):
            self.class_probs[sample[1]] += 1.0
            self.labels.append(sample[1])
            self.identifiers.append(sample[2]+'_'+str(sample[3])+'_'+str(sample[1]))
            self.device_wise[self.file_to_device[sample[2]]].append(sample)

        if self.train_flag:
            print("TRAIN DETAILS")
        else:
            print("TEST DETAILS")
         
        print("CLASSWISE SAMPLE COUNTS:", self.class_probs)
        print("Device to ID", device_to_id)
        for idx in range(device_id):
            print("DEVICE ID", idx, "size", len(self.device_wise[idx]))
        self.class_probs = self.class_probs / sum(self.class_probs)
        print("CLASSWISE PROBS", self.class_probs)
        print("LEN AUDIO DATA", len(self.audio_data))

    def new_augment(self, scale=1):

        # augment normal
        aug_nos = scale*len(self.classwise_cycle_list[0]) - len(self.classwise_cycle_list[0])
        for idx in range(aug_nos):
            # normal_i + normal_j
            i = random.randint(0, len(self.classwise_cycle_list[0])-1)
            j = random.randint(0, len(self.classwise_cycle_list[0])-1)
            normal_i = self.classwise_cycle_list[0][i]
            normal_j = self.classwise_cycle_list[0][j]
            new_sample = np.concatenate([normal_i[0], normal_j[0]])
            self.cycle_list.append((new_sample, 0, normal_i[2]+'-'+normal_j[2],
                idx, 0))
            self.filenames_with_labels.append(normal_i[2]+'-'+normal_j[2]+'_'+str(idx)+'_0')
        
        # augment crackle
        aug_nos = scale*len(self.classwise_cycle_list[0]) - len(self.classwise_cycle_list[1])
        for idx in range(aug_nos):
            aug_prob = random.random()

            if aug_prob < 0.6:
                # crackle_i + crackle_j
                i = random.randint(0, len(self.classwise_cycle_list[1])-1)
                j = random.randint(0, len(self.classwise_cycle_list[1])-1)
                sample_i = self.classwise_cycle_list[1][i]
                sample_j = self.classwise_cycle_list[1][j]
            elif aug_prob >= 0.6 and aug_prob < 0.8:
                # crackle_i + normal_j
                i = random.randint(0, len(self.classwise_cycle_list[1])-1)
                j = random.randint(0, len(self.classwise_cycle_list[0])-1)
                sample_i = self.classwise_cycle_list[1][i]
                sample_j = self.classwise_cycle_list[0][j]
            else:
                # normal_i + crackle_j
                i = random.randint(0, len(self.classwise_cycle_list[0])-1)
                j = random.randint(0, len(self.classwise_cycle_list[1])-1)
                sample_i = self.classwise_cycle_list[0][i]
                sample_j = self.classwise_cycle_list[1][j]

            new_sample = np.concatenate([sample_i[0], sample_j[0]])
            self.cycle_list.append((new_sample, 1, sample_i[2]+'-'+sample_j[2],
                idx, 0))
            self.filenames_with_labels.append(sample_i[2]+'-'+sample_j[2]+'_'+str(idx)+'_1')
        
        # augment wheeze
        aug_nos = scale*len(self.classwise_cycle_list[0]) - len(self.classwise_cycle_list[2])
        for idx in range(aug_nos):
            aug_prob = random.random()

            if aug_prob < 0.6:
                # wheeze_i + wheeze_j
                i = random.randint(0, len(self.classwise_cycle_list[2])-1)
                j = random.randint(0, len(self.classwise_cycle_list[2])-1)
                sample_i = self.classwise_cycle_list[2][i]
                sample_j = self.classwise_cycle_list[2][j]
            elif aug_prob >= 0.6 and aug_prob < 0.8:
                # wheeze_i + normal_j
                i = random.randint(0, len(self.classwise_cycle_list[2])-1)
                j = random.randint(0, len(self.classwise_cycle_list[0])-1)
                sample_i = self.classwise_cycle_list[2][i]
                sample_j = self.classwise_cycle_list[0][j]
            else:
                # normal_i + wheeze_j
                i = random.randint(0, len(self.classwise_cycle_list[0])-1)
                j = random.randint(0, len(self.classwise_cycle_list[2])-1)
                sample_i = self.classwise_cycle_list[0][i]
                sample_j = self.classwise_cycle_list[2][j]

            new_sample = np.concatenate([sample_i[0], sample_j[0]])
            self.cycle_list.append((new_sample, 2, sample_i[2]+'-'+sample_j[2],
                idx, 0))
            self.filenames_with_labels.append(sample_i[2]+'-'+sample_j[2]+'_'+str(idx)+'_2')

        # augment both
        aug_nos = scale*len(self.classwise_cycle_list[0]) - len(self.classwise_cycle_list[3])
        for idx in range(aug_nos):
            aug_prob = random.random()

            if aug_prob < 0.5:
                # both_i + both_j
                i = random.randint(0, len(self.classwise_cycle_list[3])-1)
                j = random.randint(0, len(self.classwise_cycle_list[3])-1)
                sample_i = self.classwise_cycle_list[3][i]
                sample_j = self.classwise_cycle_list[3][j]
            elif aug_prob >= 0.5 and aug_prob < 0.7:
                # crackle_i + wheeze_j
                i = random.randint(0, len(self.classwise_cycle_list[1])-1)
                j = random.randint(0, len(self.classwise_cycle_list[2])-1)
                sample_i = self.classwise_cycle_list[1][i]
                sample_j = self.classwise_cycle_list[2][j]
            elif aug_prob >=0.7 and aug_prob < 0.8:
                # wheeze_i + crackle_j
                i = random.randint(0, len(self.classwise_cycle_list[2])-1)
                j = random.randint(0, len(self.classwise_cycle_list[1])-1)
                sample_i = self.classwise_cycle_list[2][i]
                sample_j = self.classwise_cycle_list[1][j]
            elif aug_prob >=0.8 and aug_prob < 0.9:
                # both_i + normal_j
                i = random.randint(0, len(self.classwise_cycle_list[3])-1)
                j = random.randint(0, len(self.classwise_cycle_list[0])-1)
                sample_i = self.classwise_cycle_list[3][i]
                sample_j = self.classwise_cycle_list[0][j]
            else:
                # normal_i + both_j
                i = random.randint(0, len(self.classwise_cycle_list[0])-1)
                j = random.randint(0, len(self.classwise_cycle_list[3])-1)
                sample_i = self.classwise_cycle_list[0][i]
                sample_j = self.classwise_cycle_list[3][j]

            new_sample = np.concatenate([sample_i[0], sample_j[0]])
            self.cycle_list.append((new_sample, 3, sample_i[2]+'-'+sample_j[2],
                idx, 0))
            self.filenames_with_labels.append(sample_i[2]+'-'+sample_j[2]+'_'+str(idx)+'_3')

    def __getitem__(self, index):

        audio = self.audio_data[index][0]
        
        aug_prob = random.random()
        if self.train_flag and aug_prob > 0.5:
            # apply augmentation to audio
            audio = gen_augmented(audio, self.sample_rate)

            # pad incase smaller than desired length
            audio = split_and_pad([audio, 0,0,0,0], self.desired_length, self.sample_rate, types=1)[0][0]
            
        # roll audio sample
        roll_prob = random.random()
        if self.train_flag and roll_prob > 0.5:
            audio = rollAudio(audio)
        
        # convert audio signal to spectrogram
        # spectrograms resized to 3x of original size
        audio_image = cv2.cvtColor(create_mel_raw(audio, self.sample_rate, f_max=self.f_max, 
            n_mels=self.n_mels, nfft=self.nfft, hop=self.hop, resz=3), cv2.COLOR_BGR2RGB)

        # blank region clipping
        audio_raw_gray = cv2.cvtColor(create_mel_raw(audio, self.sample_rate, f_max=self.f_max, 
            n_mels=self.n_mels, nfft=self.nfft, hop=self.hop), cv2.COLOR_BGR2GRAY)

        audio_raw_gray[audio_raw_gray < 10] = 0
        for row in range(audio_raw_gray.shape[0]):
            black_percent = len(np.where(audio_raw_gray[row,:]==0)[0])/len(audio_raw_gray[row,:])
            if black_percent < 0.80:
                break

        if (row+1)*3 < audio_image.shape[0]:
            audio_image = audio_image[(row+1)*3:, :, :]
        audio_image = cv2.resize(audio_image, (audio_image.shape[1], self.n_mels*3), interpolation=cv2.INTER_LINEAR)

        if self.dump_images:
            save_images((audio_image, self.audio_data[index][2], self.audio_data[index][3], 
                self.audio_data[index][5], self.audio_data[index][1]), self.train_flag)

        # label
        label = self.audio_data[index][1]

        # apply image transform 
        if self.input_transform is not None:
            audio_image = self.input_transform(audio_image)

        return audio_image, label

    def __len__(self):
        return len(self.audio_data)
