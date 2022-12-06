# from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error
import torch
import matplotlib.pyplot as plt
import numpy as np
from config import args
from tensorboardX import SummaryWriter
import time
from torchvision.utils import save_image


def save_img(input, output, comp_B, mask, save_path):
    input = input.detach().cpu().numpy()[0, 0]
    output = output.detach().cpu().numpy()[0, 0]
    mask = mask.detach().cpu().numpy()[0, 0]
    comp_B = comp_B.detach().cpu().numpy()[0, 0]

    input_tline = input[0, :, :]
    input_xline = input[:, 0, :]
    input_iline = input[:, :, 0]

    output_tline = output[0, :, :]
    output_xline = output[:, 0, :]
    output_iline = output[:, :, 0]

    comp_B_tline = comp_B[0, :, :]
    comp_B_xline = comp_B[:, 0, :]
    comp_B_iline = comp_B[:, :, 0]

    mask_tline = mask[0, :, :]
    mask_xline = mask[:, 0, :]
    mask_iline = mask[:, :, 0]

    imgs = [input_tline, input_xline, input_iline, output_tline, output_xline, output_iline, mask_tline * input_tline,
            mask_xline * input_xline,
            mask_iline * input_iline, comp_B_tline, comp_B_xline, comp_B_iline]

    for i in range(len(imgs)):
        imgs[i] = plt.get_cmap('seismic')(imgs[i])[:, :, :-1].transpose((2, 0, 1))[None, :, :, :]

    imgs = torch.from_numpy(np.concatenate(imgs, axis=0))
    save_image(imgs, save_path + '_0_.png', nrow=3)
    # ==============================================
    input_tline = input[-1, :, :]
    input_xline = input[:, -1, :]
    input_iline = input[:, :, -1]

    output_tline = output[-1, :, :]
    output_xline = output[:, -1, :]
    output_iline = output[:, :, -1]

    comp_B_tline = comp_B[-1, :, :]
    comp_B_xline = comp_B[:, -1, :]
    comp_B_iline = comp_B[:, :, -1]

    mask_tline = mask[-1, :, :]
    mask_xline = mask[:, -1, :]
    mask_iline = mask[:, :, -1]

    imgs = [input_tline, input_xline, input_iline, output_tline, output_xline, output_iline, mask_tline * input_tline,
            mask_xline * input_xline,
            mask_iline * input_iline, comp_B_tline, comp_B_xline, comp_B_iline]

    for i in range(len(imgs)):
        imgs[i] = plt.get_cmap('seismic')(imgs[i])[:, :, :-1].transpose((2, 0, 1))[None, :, :, :]

    imgs = torch.from_numpy(np.concatenate(imgs, axis=0))
    save_image(imgs, save_path + '-1_.png', nrow=3)

    input_tline = input[64, :, :]
    input_xline = input[:, 64, :]
    input_iline = input[:, :, 64]

    output_tline = output[64, :, :]
    output_xline = output[:, 64, :]
    output_iline = output[:, :, 64]

    comp_B_tline = comp_B[64, :, :]
    comp_B_xline = comp_B[:, 64, :]
    comp_B_iline = comp_B[:, :, 64]

    mask_tline = mask[64, :, :]
    mask_xline = mask[:, 64, :]
    mask_iline = mask[:, :, 64]

    imgs = [input_tline, input_xline, input_iline, output_tline, output_xline, output_iline, mask_tline * input_tline,
            mask_xline * input_xline,
            mask_iline * input_iline, comp_B_tline, comp_B_xline, comp_B_iline]

    for i in range(len(imgs)):
        imgs[i] = plt.get_cmap('seismic')(imgs[i])[:, :, :-1].transpose((2, 0, 1))[None, :, :, :]

    imgs = torch.from_numpy(np.concatenate(imgs, axis=0))
    save_image(imgs, save_path + '_64_.png', nrow=3)


class lossState():
    def __init__(self, print_freq):
        self.print_frep = print_freq
        self.g_gan_loss_3d = []
        self.d_loss_3d = []
        self.tce_loss = []
        self.g_gan_loss_2d = []
        self.d_loss_2d_t = []
        self.d_loss_2d_ix = []
        self.start_time = time.time()

    def update(self, g_gan_loss_3d, d_loss_3d, tce_loss, g_gan_loss_2d, d_loss_2d_t, d_loss_2d_ix, idx):
        self.g_gan_loss_3d.append(g_gan_loss_3d)
        self.d_loss_3d.append(d_loss_3d)
        self.tce_loss.append(tce_loss)
        self.g_gan_loss_2d.append(g_gan_loss_2d)
        self.d_loss_2d_t.append(d_loss_2d_t)
        self.d_loss_2d_ix.append(d_loss_2d_ix)

        if idx % self.print_frep == 0:
            g_gan_loss_3d = np.mean(self.g_gan_loss_3d)
            d_loss_3d = np.mean(self.d_loss_3d)
            tce_loss = np.mean(self.tce_loss)
            g_gan_loss_2d = np.mean(self.g_gan_loss_2d)
            d_loss_2d_t = np.mean(self.d_loss_2d_t)
            d_loss_2d_ix = np.mean(self.d_loss_2d_ix)

            print("==================Iteration:%d, time taken: %.2f===================" % (
                idx, time.time() - self.start_time))
            print("g_gan_loss_3d:%.4f, g_gan_loss_2d:%.4f, tce_loss:%.4f" % (
                g_gan_loss_3d, g_gan_loss_2d, tce_loss))
            print("d_loss_3d:%.4f, d_loss_2d_t:%.4f, d_loss_2d_ix:%.4f" % (
                d_loss_3d, d_loss_2d_t, d_loss_2d_ix))
            print('=====================================================================')

            self.start_time = time.time()
            self.g_gan_loss_3d = []
            self.d_loss_3d = []
            self.tce_loss = []
            self.g_gan_loss_2d = []
            self.d_loss_2d_t = []
            self.d_loss_2d_ix = []
