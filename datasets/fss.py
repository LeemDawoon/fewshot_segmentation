import random
import os
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils import data
from torchvision.transforms import (
    Compose,
    ToTensor,
)


def _crop_image(image):
    output = image.copy()
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None, None, None

    len_list = [len(c) for c in contours]
    max_idx = len_list.index(np.max(len_list))
    cnt = contours[max_idx]
    x, y, w, h = cv2.boundingRect(cnt)  # 좌상단이 0, 0, x=shape[1], y=shape[0]
    return x, y, w, h

def _adjust_crop_ragne(ymin, ymax, xmin, xmax, img_shape):
    if ymin < 0:
        margin = abs(ymin)
        ymin = 0
        ymax += margin
    if ymax > img_shape[0]:
        margin = ymax - img_shape[0]
        ymax = img_shape[0]
        ymin -= margin
    if xmin < 0:
        margin = abs(xmin)
        xmin = 0
        xmax += margin
    if xmax > img_shape[1]:
        margin = xmax - img_shape[1]
        xmax = img_shape[1]
        xmin -= margin
    return ymin, ymax, xmin, xmax


def fss_preprocess(img, mask, size):
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    img = img.astype('uint8')

    ####################################################################################################################
    # 마스크 전처리
    ####################################################################################################################
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    canvas_mask[cent - hy:cent + hy, cent - hx:cent + hx] += mask[ymin:ymax, xmin:xmax]
    canvas_mask = cv2.resize(canvas_mask, (size, size), interpolation=cv2.INTER_CUBIC)

    # disc
    ind_disc = np.where(canvas_mask <= 180)
    disc_mask = np.zeros(canvas_mask.shape)
    disc_mask[ind_disc[0], ind_disc[1]] = 1
    disc_mask = disc_mask.astype('uint8')
    return img, disc_mask


def getMask(label, class_id, class_ids):
    """
    Generate FG/BG mask from the segmentation mask

    Args:
        label:
            semantic mask
        scribble:
            scribble mask
        class_id:
            semantic class of interest
        class_ids:
            all class id in this episode
    """
    # Dense Mask
    fg_mask = torch.where(label == class_id, torch.ones_like(label), torch.zeros_like(label))
    bg_mask = torch.where(label != class_id, torch.ones_like(label), torch.zeros_like(label))
    for class_id in class_ids:
        bg_mask[label == class_id] = 0


    return {'fg_mask': fg_mask,
            'bg_mask': bg_mask}



class FSS(data.Dataset):
    def __init__(
        self,
        class_list,
        data_root='',
        augmentation=None,
        image_size=224,
        n_way=1,
        k_shot=5, 
        n_class_samples=10,
        ):

        # self.images = xs
        # self.masks = ys
        self.class_list = class_list
        self.data_root = data_root
        self.augmentation = augmentation
        self.image_size = image_size
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_class_samples = n_class_samples

    def __getitem__(self, idx):

        # indx_c = random.sample(range(0, len(self.class_list)), self.n_way)
        if idx not in list(range(0, len(self.class_list))):
            raise Exception('>>> here ...')
            
        indx_c = [idx]
        # print('indx_c', indx_c)
        indx_s = random.sample(range(1, self.n_class_samples+1), self.n_class_samples) # 셔플 느낌.

        support    = np.zeros([self.n_way, self.k_shot, 3, self.image_size, self.image_size], dtype = np.float32)
        smasks_fg  = np.zeros([self.n_way, self.k_shot, 1, self.image_size, self.image_size], dtype = np.float32)
        smasks_bg  = np.zeros([self.n_way, self.k_shot, 1, self.image_size, self.image_size], dtype = np.float32)
        # smasks_fg  = np.zeros([self.n_way, self.k_shot, 56,        56,        1], dtype = np.float32)
        # smasks_bg  = np.zeros([self.n_way, self.k_shot, 56,        56,        1], dtype = np.float32)
        query   = np.zeros([self.n_way, 3, self.image_size, self.image_size], dtype = np.float32)      
        qmask   = np.zeros([self.n_way, 1, self.image_size, self.image_size], dtype = np.float32)  

        img_path_list = []
        msk_path_list = []
        for idx in range(len(indx_c)):
            # print(self.data_root+ '/' + self.class_list[indx_c[idx]] + '/' + str(indx_s[0]) + '.jpg' )
            for idy in range(self.k_shot): # For support set 
                s_img_path = self.data_root+ '/' + self.class_list[indx_c[idx]] + '/' + str(indx_s[idy]) + '.jpg'
                s_msk_path = self.data_root+ '/' + self.class_list[indx_c[idx]] + '/' + str(indx_s[idy]) + '.png'
                img_path_list.append(s_img_path)
                msk_path_list.append(s_msk_path)
                s_img = cv2.imread(s_img_path)
                s_msk = cv2.imread(s_msk_path)
                s_img = cv2.resize(s_img,(self.image_size, self.image_size))
                s_msk = cv2.resize(s_msk,(self.image_size, self.image_size))
                # s_msk = cv2.resize(s_msk,(56,        56))        
                s_msk = s_msk /255.
                s_msk_fg = np.where(s_msk > 0.5, 1., 0.)
                s_msk_bg = np.where(s_msk <= 0.5, 1., 0.)
                support[idx, idy] = s_img.transpose(2, 0, 1)
                # print(support[idx, idy].shape) # (3, 224, 224)
                smasks_fg[idx, idy]  = s_msk_fg[:, :, 0:1].transpose(2, 0, 1)
                smasks_bg[idx, idy]  = s_msk_bg[:, :, 0:1].transpose(2, 0, 1)

            for idy in range(1): # For query set consider 1 sample per class
                q_img_path = self.data_root+ '/' + self.class_list[indx_c[idx]] + '/' + str(indx_s[idy+self.k_shot]) + '.jpg'
                q_msk_path = self.data_root+ '/' + self.class_list[indx_c[idx]] + '/' + str(indx_s[idy+self.k_shot]) + '.png'
                img_path_list.append(q_img_path)
                msk_path_list.append(q_msk_path)
                q_img = cv2.imread(q_img_path)
                q_msk = cv2.imread(q_msk_path)
                q_img = cv2.resize(q_img,(self.image_size, self.image_size))
                q_msk = cv2.resize(q_msk,(self.image_size, self.image_size))        
                q_msk = q_msk /255.
                q_msk = np.where(q_msk > 0.5, 1., 0.)
                query[idx] = q_img.transpose(2, 0, 1)
                qmask[idx] = q_msk[:, :, 0:1].transpose(2, 0, 1)   

        support = support /255.
        query   = query   /255.

        # support_mask = [[getMask(support_labels[way][shot], class_ids[way], class_ids) for shot in range(n_shots)] for way in range(n_ways)]
        # print('>>> support.shape', support.shape) # support.shape (1, 5, 3, 224, 224)
        # print(q_img_path)
        return support, smasks_fg, smasks_bg, query, qmask, [img_path_list, msk_path_list]

    def __len__(self):
        return len(self.class_list)


    
# import glob
# import random
# import pandas as pd
# SEED = 1337

if __name__ == "__main__":
    pass
