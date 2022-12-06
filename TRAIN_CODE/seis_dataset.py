from PIL import Image, ImageFile
import torchvision.transforms as transforms
import torch.utils.data as data
from skimage import transform, filters
from modules import task
import random
import numpy as np
import os
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as visF


def randomCrop(data, size=(128, 128, 128)):
    shape = np.array(data.shape)
    lim = shape - size
    w = random.randint(0, lim[0])
    h = random.randint(0, lim[1])
    c = random.randint(0, lim[2])
    return data[w:w + size[0], h:h + size[1], c:c + size[2]]


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / (_range+0.00001)

def z_score_clip(data, clp_s):
    z = (data - np.mean(data)) / np.std(data)
    return normalization(np.clip(z, a_min=-clp_s, a_max=clp_s))


def RandomNoise(seismic):
    if random.random() < 0.15:  return seismic
    s_max, s_min = torch.max(seismic), torch.min(seismic)
    scale = random.random() * (s_max - s_min) * 0.05
    noise = torch.normal(mean=0.0, std=scale, size=seismic.shape)
    if random.randint(0, 1):
        sigma = random.randint(1, 3)
        noise = visF.gaussian_blur(noise, kernel_size=sigma * 2 + 1, sigma=sigma)
    seismic = seismic + noise
    seismic = torch.clip(seismic, min=s_min, max=s_max)
    return seismic


def RandomHorizontalFlipCoord(seismic, p=0.5):
    if random.random() < p:
        return visF.hflip(seismic)
    return seismic

def RandomVerticalFlipCoord(seismic, p=0.5):
    if random.random() < p:
        return visF.vflip(seismic)
    return seismic

def RandomRotateCoord(seismic, p=0.5):
    if random.random() < p:
        return seismic.permute((0, 1, 3, 2))
    return seismic


class seis_dataset(data.Dataset):
    def __init__(self, args):
        self.seis_path = args.train_path
        self.seis_list = os.listdir(self.seis_path)
        self.seis_size = len(self.seis_list)
        self.all_step = args.train_step * args.batch_size
        self.train_datashape = np.array(args.data_shape, dtype=np.int32)

    def __len__(self):
        return self.all_step

    def load_seis(self, index):
        seis = np.load(os.path.join(self.seis_path, self.seis_list[index]))

        clp_s = random.random() * 12 + 3.2

        seis = z_score_clip(seis, clp_s)
        return seis

    def __getitem__(self, index):
        # load image
        seis = self.load_seis(index % self.seis_size)
        seis = torch.from_numpy(seis).float()
        if not (self.train_datashape == np.array(seis.shape)).all():
            seis_shape = np.clip(np.array(seis.shape), a_min=np.min(self.train_datashape), a_max=np.inf).astype(np.int32)
            if not (seis_shape == np.array(seis.shape)).all():
                seis = F.interpolate(seis[None, None], size=seis_shape.tolist(), mode='trilinear', align_corners=True)[
                    0, 0]
            seis = randomCrop(seis, self.train_datashape)
        mask = torch.from_numpy(self.load_mask()).float()

        if random.randint(0, 3) == 0:
            scale1 = random.randint(128, 200)
            scale2 = random.randint(128, 200)
            scale3 = random.randint(128, 200)
            seis = F.interpolate(seis[None, None], (scale1, scale2, scale3), mode='trilinear', align_corners=True)[0, 0]
            seis = randomCrop(seis)

        seis = seis[None]
        #seis = RandomNoise(seis)
        #seis = RandomHorizontalFlipCoord(seis)
        seis = RandomVerticalFlipCoord(seis)
        seis = RandomRotateCoord(seis)
        return seis, mask

    def load_mask(self):
        shape = [1]+self.train_datashape.tolist()

        mask_type = random.randint(0, 5)

        if mask_type == 0:
            return task.center_mask(shape)
        if mask_type == 1:
            return task.random_regular_mask(shape)
        if mask_type == 2:
            return task.slices_mask(shape)
        if mask_type == 3:
            return task.mix_mask(shape)
        if mask_type == 4:
            return task.cube_mask(shape)
        else:
            return task.mid_mask(shape)
