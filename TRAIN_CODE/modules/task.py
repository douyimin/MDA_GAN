import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from random import randint
import numpy as np
import cv2
from PIL import Image
import random


###################################################################
# 这里写的很乱，但是它是work的，我懒得改了
# The writing here is messy, but it's WORK, I'm too lazy to change it
###################################################################


def random_regular_mask(shape):
    mask = np.ones(shape)
    s = shape
    N_mask = random.randint(6, 12)
    limx = s[1] - s[1] / (N_mask + 1)
    limy = s[2] - s[2] / (N_mask + 1)
    limz = s[3] - s[3] / (N_mask + 1)
    for _ in range(N_mask):
        x = random.randint(0, int(limx))
        y = random.randint(0, int(limy))
        z = random.randint(0, int(limz))
        range_x = x + random.randint(int(s[1] / (N_mask + 7)), int(s[1] - x))
        range_y = y + random.randint(int(s[2] / (N_mask + 7)), int(s[2] - y))
        range_z = z + random.randint(int(s[3] / (N_mask + 7)), int(s[3] - z))
        mask[:, int(x):int(range_x), int(y):int(range_y), int(z):int(range_z)] = 0
    return 1 - mask


def center_mask(shape):
    mask = np.ones(shape)
    size = shape

    if random.randint(0, 1):

        ty = np.random.randint(75, 125)
        tx = np.random.randint(75, 125)

        tyy = 128 - ty - 1
        txx = 128 - tx - 1

        bty = np.random.randint(0, tyy)
        btx = np.random.randint(0, txx)

        if random.randint(0, 1):
            mask[:, :, btx:btx + tx, :] = 0
        else:
            mask[:, :, :, bty:bty + ty] = 0
        return mask
    else:
        ty = np.random.randint(1, 30)
        tx = np.random.randint(1, 30)
        tyy = 128 - ty - 1
        txx = 128 - tx - 1
        bty = np.random.randint(0, tyy)
        btx = np.random.randint(0, txx)

        rand = random.randint(0, 2)
        if rand == 0:
            mask[:, :, btx:btx + tx, :] = 0
        elif rand == 1:
            mask[:, :, :, bty:bty + ty] = 0
        else:
            mask[:, :, btx:btx + tx, :] = 0
            mask[:, :, :, bty:bty + ty] = 0
        return 1 - mask


def mid_mask(shape):
    mask = np.ones(shape)
    size = shape
    x = random.randint(2, 32)
    y = random.randint(2, 32)
    z = random.randint(2, 32)
    t = random.randint(0, 6)
    if t == 0:
        if random.randint(0, 1):
            mask[:, :, y:, z:] = 0
        else:
            mask[:, :, :-y, :-z] = 0
    if t == 1:
        if random.randint(0, 1):
            mask[:, x:, :, z:] = 0
        else:
            mask[:, :-x, :, :-z] = 0
    if t == 2:
        if random.randint(0, 1):
            mask[:, x:, y:, :] = 0
        else:
            mask[:, :-x, :-y, :] = 0
    if t == 3:
        if random.randint(0, 1):
            mask[:, :, :, z:] = 0
        else:
            mask[:, :, :, :-z] = 0
    if t == 4:
        if random.randint(0, 1):
            mask[:, :, :, z:] = 0
        else:
            mask[:, :, :, :-z] = 0
    if t == 5:
        if random.randint(0, 1):
            mask[:, :, y:, :] = 0
        else:
            mask[:, :, :-y, :] = 0
    if t == 6:
        if random.randint(0, 1):
            mask[:, x:, y:, z:] = 0
        else:
            mask[:, :-x, :-y, :-z] = 0
    return mask


def slices_mask(shape):
    mask = np.ones(shape)
    size = shape
    if random.randint(0, 1):
        prop = random.randint(50, 95) / 100.
        # x = random.sample(range(0, size[1]), int((size[3]) * prop))
        y = random.sample(range(0, size[2]), int((size[2]) * prop))
        z = random.sample(range(0, size[3]), int((size[3]) * prop))
        rand = random.randint(0, 2)
        if rand == 0:
            for i in z: mask[:, :, :, i] = 0
        elif rand == 1:
            for i in y: mask[:, :, i, :] = 0
        else:
            for i in y: mask[:, :, i, :] = 0
            for i in z: mask[:, :, :, i] = 0
        return mask
    else:
        prop = random.randint(30, 95) / 100.
        x = random.sample(range(0, size[1]), int((size[1]) * prop))
        y = random.sample(range(0, size[2]), int((size[2]) * prop))
        z = random.sample(range(0, size[3]), int((size[3]) * prop))

        for i in x: mask[:, i, :, :] = 0
        for i in y: mask[:, :, i, :] = 0
        for i in z: mask[:, :, :, i] = 0
        return mask


def mix_mask(shape):
    mask = np.ones(shape)
    size = shape

    y = np.random.randint(0, 128, random.randint(100, 128))
    z = np.random.randint(0, 128, random.randint(100, 128))

    for i in y: mask[:, :, i, :] = 0
    for i in z: mask[:, :, :, i] = 0

    if random.randint(0, 1):
        ty = np.random.randint(50, 95)
        tx = np.random.randint(50, 95)

        tyy = 128 - ty - 1
        txx = 128 - tx - 1

        bty = np.random.randint(0, tyy)
        btx = np.random.randint(0, txx)

        r = random.randint(0, 2)
        if r == 0:
            mask[:, :, btx:btx + tx, :] = 0
        elif r == 1:
            mask[:, :, :, bty:bty + ty] = 0
        else:
            mask[:, :, btx:btx + tx, :] = 0
            mask[:, :, :, bty:bty + ty] = 0

        return mask
    else:
        s = shape
        N_mask = random.randint(6, 12)
        limx = s[1] - s[1] / (N_mask + 1)
        limy = s[2] - s[2] / (N_mask + 1)
        limz = s[3] - s[3] / (N_mask + 1)
        for _ in range(N_mask):
            x = random.randint(0, int(limx))
            y = random.randint(0, int(limy))
            z = random.randint(0, int(limz))
            range_x = x + random.randint(int(s[1] / (N_mask + 7)), int(s[1] - x))
            range_y = y + random.randint(int(s[2] / (N_mask + 7)), int(s[2] - y))
            range_z = z + random.randint(int(s[3] / (N_mask + 7)), int(s[3] - z))
            mask[:, int(x):int(range_x), int(y):int(range_y), int(z):int(range_z)] = 0
        return mask


def cube_mask(shape):
    rand = random.randint(0, 3)
    if rand == 0:
        cube_size = random.randint(3, 20)
        mask = np.ones(shape)
        prop = random.randint(2, 5)
        count = int((128 // cube_size) ** 3 * prop)
        for i in range(count):
            x = random.randint(0, 128 - cube_size)
            y = random.randint(0, 128 - cube_size)
            z = random.randint(0, 128 - cube_size)
            mask[:, z:z + cube_size, x:x + cube_size, y:y + cube_size] = 0
        return mask

    elif rand == 1:
        return mid_mask(shape)

    elif rand == 2:
        scale = random.randint(4, 20)
        mask = np.zeros(shape)
        mask_l = int(128 // scale)
        if random.randint(0, 1):
            for i in range(0, mask_l):
                mask[:, i * scale, :, :] = 1
        else:
            if random.randint(0, 1):
                for i in range(0, mask_l):
                    mask[:, :, i * scale, :] = 1
            else:
                for i in range(0, mask_l):
                    mask[:, :, :, i * scale] = 1
        return mask

    elif rand == 3:
        return single_mask(shape)


def single_mask(shape):
    mask = np.zeros(shape)
    prop = random.randint(10, 95) / 100.
    s = random.randint(18, 110)
    if random.randint(0, 1):
        mask[:, :, s, :] = 1
        x = random.sample(range(0, 128), int(128 * prop))
        for i in x: mask[:, :, :, i] = 0
    else:
        mask[:, :, :, s] = 1
        x = random.sample(range(0, 128), int(128 * prop))
        for i in x: mask[:, :, i, :] = 0
    return mask
