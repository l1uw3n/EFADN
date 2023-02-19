# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 11:30 上午
# @Author  : liuwen
# @File    : Load_Dataset.py
# @Software: PyCharm
import torch
import random
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import netCDF4 
import numpy as np
import scipy.ndimage
import gc

def random_rot_flip(sst_sla, mask, img, grid):
    k = np.random.randint(0, 4)
    sst_sla = np.rot90(sst_sla, k)
    mask = np.rot90(mask, k)
    img = np.rot90(img, k)
    grid = np.rot90(grid, k)
    axis = np.random.randint(0, 2)
    sst_sla = np.flip(sst_sla, axis=axis).copy()
    mask = np.flip(mask, axis=axis).copy()
    img = np.flip(img, axis=axis).copy()
    grid = np.flip(grid, axis=axis).copy()
    return sst_sla, mask, img, grid


class RandomGenerator(object):
    def __init__(self):
        self.output_size = 0
    def __call__(self, sample):
        sst_sla, mask, img, grid = sample['sst_sla'], sample['mask'], sample['img'], sample['grid']
        # if random.random() > 0.5:
            # sst_sla, mask, img, grid = random_rot_flip(sst_sla, mask, img, grid) #不做扩充
        grid=grid[np.newaxis, :, :]
        sample = {'sst_sla': sst_sla, 'mask': mask, 'img':img, 'grid':grid}
        return sample

class ValGenerator(object):
    def __init__(self):
        self.output_size = 0
    def __call__(self, sample):
        sst_sla, mask, img, grid = sample['sst_sla'], sample['mask'], sample['img'], sample['grid']
        grid=grid[np.newaxis, :, :]
        sample = {'sst_sla': sst_sla, 'mask': mask, 'img':img, 'grid':grid}
        return sample


class ImageToImage2D(Dataset):
    def __init__(self, joint_transform: Callable = None) -> None:
        #南美
        e2a2013 = np.load("./5_front_eddy_mask_e2_2013.npy", allow_pickle=True).item()
        e2a2014 = np.load("./5_front_eddy_mask_e2_2014.npy", allow_pickle=True).item()
        print(1)
        e3a2014 = np.load("./5_front_eddy_mask_e3_2014.npy", allow_pickle=True).item()
        print(2)
        w1a2014 = np.load("./5_front_eddy_mask_w1_2014.npy", allow_pickle=True).item()

        sst_sla = []
        for iter in range(len(e2a2013["sst"])):
            sst_sla.append(np.dstack([scipy.ndimage.zoom(e2a2013["sst"][iter], 0.6), scipy.ndimage.zoom(e2a2013["sla"][iter], 3)]))
        for iter in range(len(e2a2014["sst"])):
            sst_sla.append(np.dstack([scipy.ndimage.zoom(e2a2014["sst"][iter], 0.6), scipy.ndimage.zoom(e2a2014["sla"][iter], 3)]))
        for iter in range(len(e3a2014["sst"])):
            sst_sla.append(np.dstack([scipy.ndimage.zoom(e3a2014["sst"][iter], 0.6), scipy.ndimage.zoom(e3a2014["sla"][iter], 3)]))
        for iter in range(len(w1a2014["sst"])):
            sst_sla.append(np.dstack([scipy.ndimage.zoom(w1a2014["sst"][iter], 0.6), scipy.ndimage.zoom(w1a2014["sla"][iter], 3)]))

        sst_sla = np.array(sst_sla)
        self.sst_sla = np.array(sst_sla)
        self.sst_sla = self.sst_sla.astype(np.float32)
        
        
        print("sst_sla down",self.sst_sla.shape)
        self.img = np.array(np.vstack([e2a2013["img"],e2a2014["img"],e3a2014["img"],w1a2014["img"]]))
        # self.img = np.array(np.vstack([e3a2014["img"]]))
        self.img = self.img.astype(np.float32)

        print("img down",self.img.shape)
        self.mask = np.array(np.vstack([e2a2013["mask"],e2a2014["mask"],e3a2014["mask"],w1a2014["mask"]]))
        print("mask down",self.mask.shape)
        #e2a2013 e2a2014 e3a2014 w1a2014
        gridarr = []
        for iter in range(len(e2a2013["grid"])):
            gridarr.append(scipy.ndimage.zoom(e2a2013["grid"][iter], 0.6))
        for iter in range(len(e2a2014["grid"])):
            gridarr.append(scipy.ndimage.zoom(e2a2014["grid"][iter], 0.6))
        for iter in range(len(e3a2014["grid"])):
            gridarr.append(scipy.ndimage.zoom(e3a2014["grid"][iter], 0.6))
        for iter in range(len(w1a2014["grid"])):
            gridarr.append(scipy.ndimage.zoom(w1a2014["grid"][iter], 0.6))
        self.grid = np.array(gridarr)
        print("grid down",self.grid.shape)
        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))
        print(self.sst_sla.shape,"$$$$$$$$$$$$$$$$$$$$$$")
    def __len__(self):
        return len(self.sst_sla)
    def __getitem__(self, idx):
        sample = {'sst_sla': self.sst_sla[idx], 'mask': self.mask[idx], 'img':self.img[idx], 'grid':self.grid[idx]}
        if self.joint_transform:
            sample = self.joint_transform(sample)
        return {
            'sst_sla': torch.as_tensor(sample['sst_sla']).permute(2, 0, 1).float().contiguous(),
            'mask': torch.as_tensor(sample['mask']).long().contiguous(),
            'img': torch.as_tensor(sample['img']).permute(2, 0, 1).float().contiguous(),
            'grid': torch.as_tensor(sample['grid']).float().contiguous()
        }


class ImageToImage2Dval(Dataset):
    def __init__(self, joint_transform: Callable = None) -> None:

        e2a2014 = np.load("./5_front_eddy_mask_e2_2015.npy", allow_pickle=True).item()
        print(1)
        e3a2014 = np.load("./5_front_eddy_mask_e3_2015.npy", allow_pickle=True).item()
        print(2)
        w1a2014 = np.load("./5_front_eddy_mask_w1_2015.npy", allow_pickle=True).item()
        sst_sla = []
        for iter in range(0,len(e2a2014["sst"]),10):
            sst_sla.append(np.dstack([scipy.ndimage.zoom(e2a2014["sst"][iter], 0.6), scipy.ndimage.zoom(e2a2014["sla"][iter], 3)]))
        for iter in range(0,len(e3a2014["sst"]),10):
            sst_sla.append(np.dstack([scipy.ndimage.zoom(e3a2014["sst"][iter], 0.6), scipy.ndimage.zoom(e3a2014["sla"][iter], 3)]))
        for iter in range(0,len(w1a2014["sst"]),100):
            sst_sla.append(np.dstack([scipy.ndimage.zoom(w1a2014["sst"][iter], 0.6), scipy.ndimage.zoom(w1a2014["sla"][iter], 3)]))
        sst_sla = np.array(sst_sla)
        self.sst_sla = np.array(sst_sla)
        self.sst_sla = self.sst_sla.astype(np.float32)
        
        
        print("sst_sla down",self.sst_sla.shape)

        imgtem = []
        for iter in range(0,len(e2a2014["img"])):
            imgtem.append(e2a2014["img"][iter])
        for iter in range(0,len(e3a2014["img"])):
            imgtem.append(e3a2014["img"][iter])
        for iter in range(0,len(w1a2014["img"])):
            imgtem.append(w1a2014["img"][iter])
        self.img = np.array(imgtem)

        masktem = []
        for iter in range(0,len(e2a2014["mask"])):
            masktem.append(e2a2014["mask"][iter])
        for iter in range(0,len(e3a2014["mask"])):
            masktem.append(e3a2014["mask"][iter])
        for iter in range(0,len(w1a2014["mask"])):
            masktem.append(w1a2014["mask"][iter])
        self.mask = np.array(masktem)
        print("mask down",self.mask.shape)
        gridarr = []
        for iter in range(0,len(e2a2014["grid"])):
            gridarr.append(scipy.ndimage.zoom(e2a2014["grid"][iter], 0.6))
        for iter in range(0,len(e3a2014["grid"])):
            gridarr.append(scipy.ndimage.zoom(e3a2014["grid"][iter], 0.6))
        for iter in range(0,len(w1a2014["grid"])):
            gridarr.append(scipy.ndimage.zoom(w1a2014["grid"][iter], 0.6))
        self.grid = np.array(gridarr)
 
        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))
        print(self.sst_sla.shape,"$$$$$$$$$$$$$$$$$$$$$$")
    def __len__(self):
        return len(self.sst_sla)
    def __getitem__(self, idx):
        sample = {'sst_sla': self.sst_sla[idx], 'mask': self.mask[idx], 'img':self.img[idx], 'grid':self.grid[idx]}
        if self.joint_transform:
            sample = self.joint_transform(sample)
        return {
            'sst_sla': torch.as_tensor(sample['sst_sla']).permute(2, 0, 1).float().contiguous(),
            'mask': torch.as_tensor(sample['mask']).long().contiguous(),
            'img': torch.as_tensor(sample['img']).permute(2, 0, 1).float().contiguous(),
            'grid': torch.as_tensor(sample['grid']).float().contiguous()
        }

