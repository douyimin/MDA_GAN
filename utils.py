"""Copyright China University of Petroleum East China, Yimin Dou, Kewen Li

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import random

random.seed(2022)
np.random.seed(2022)


def discrete_missing(data, proportions=(0.75, 0.75, 0.75), directions=('iline', 'xline', 'tline'), mask=None):
    assert len(proportions) == len(directions)
    assert set(directions).issubset({'iline', 'xline', 'tline'})
    d_dict = dict(zip(directions, proportions))
    if mask is None: mask = np.ones_like(data)
    size = data.shape
    for direction in directions:
        proportion = d_dict[direction]
        print(f'{direction} random discrete missing proportion: {proportion * 100}%')
        if direction == 'iline':
            sample = random.sample(range(1, size[1]), int((size[1] - 1) * proportion))
            for i in sample: mask[:, i, :] = 0
        elif direction == 'xline':
            sample = random.sample(range(1, size[2]), int((size[2] - 1) * proportion))
            for i in sample: mask[:, :, i] = 0
        else:
            sample = random.sample(range(1, size[0]), int((size[0] - 1) * proportion))
            for i in sample: mask[i, :, :] = 0
    return mask


def continuous_missing(data, num_traces=(50, 50, 50), directions=('iline', 'xline', 'tline'),
                       start_missing=(20, 20, 20), mask=None):
    assert len(num_traces) == len(directions)
    assert len(num_traces) == len(start_missing)
    assert set(directions).issubset({'iline', 'xline', 'tline'})
    if mask == None: mask = np.ones_like(data)
    size = data.shape
    start_dict = dict(zip(directions, start_missing))
    traces_dict = dict(zip(directions, num_traces))
    for direction in directions:
        start = start_dict[direction]
        trace = traces_dict[direction]
        print(f'{direction} continuous missing traces: {trace}, missing start trace {start}')
        if direction == 'iline':
            assert start < size[1]
            mask[:, start:start + trace, :] = 0
        elif direction == 'xline':
            assert start < size[2]
            mask[:, :, start:start + trace] = 0
        else:
            assert start < size[0]
            mask[start:start + trace, :, :] = 0
    return mask


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def normalization_tensor(data):
    _range = torch.max(data) - torch.min(data)
    return (data - torch.min(data)) / _range


def z_score_clip(data):
    z = (data - np.mean(data)) / np.std(data)
    return normalization(np.clip(z, a_min=-3.8, a_max=3.8))


def prediction(model, data, device):
    model.eval()
    data = normalization(data)
    m1, m2, m3 = data.shape
    c1 = (np.ceil(m1 / 16) * 16).astype(np.int)
    c2 = (np.ceil(m2 / 16) * 16).astype(np.int)
    c3 = (np.ceil(m3 / 16) * 16).astype(np.int)
    input_tensor = np.zeros((c1, c2, c3), dtype=np.float32) + 0.5
    input_tensor[:m1, :m2, :m3] = data
    input_tensor = torch.from_numpy(input_tensor)[None, None, :, :, :].to(device)
    if device.type == 'cpu':
        input_tensor = input_tensor.float()
    else:
        input_tensor = input_tensor.half()
    with torch.no_grad():
        result = model(input_tensor).cpu().numpy()[0, 0, :m1, :m2, :m3]
    return result


def get_pseudo_color_img(org_slice, masked_slice, filled_slices):
    org_slice = plt.get_cmap('seismic')(z_score_clip(org_slice))[:, :, :-1]
    masked_slice = plt.get_cmap('seismic')(z_score_clip(masked_slice))[:, :, :-1]
    filled_slices = plt.get_cmap('seismic')(z_score_clip(filled_slices))[:, :, :-1]
    # grid = np.concatenate([np.ones((org_slice.shape[0], 5, 1)), np.ones((org_slice.shape[0], 5, 1)),
    #                       np.zeros((org_slice.shape[0], 5, 1))],2)
    # cat_img = np.concatenate((org_slice, grid, masked_slice, grid, filled_slices), 1)
    # return cat_img
    return org_slice, masked_slice, filled_slices


def calculate_missing_ratio(mask):
    nomasked = np.sum((mask == 1).astype(np.float32))
    masked = np.sum((mask == 0).astype(np.float32))
    return masked / (nomasked + masked)
