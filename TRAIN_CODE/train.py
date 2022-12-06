import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from model_metric import lossState
from config import args
from torch.utils.data import DataLoader

from seis_dataset import seis_dataset
from modules.D_Net import D_Net2d, D_Net3d, AdversarialLoss, TCE_loss
from modules.G_Net import G_Net
import random
from torch.cuda.amp import autocast, GradScaler
from model_metric import save_img


def get_2d_slice(gt_seis, fake_seis):
    B, C, T, I, X = gt_seis.shape
    sample_slices_t = random.sample(range(T), args.sample_slices)  # B,C,T,I,X
    sample_slices_i = random.sample(range(I), int(args.sample_slices / 2))  # default I==X bixu xiangdeng
    sample_slices_x = random.sample(range(X), int(args.sample_slices / 2))

    gt_seis_slice_t = gt_seis[:, :, sample_slices_t].permute((0, 2, 1, 3, 4)).reshape(-1, C, I, X)
    gt_seis_slice_i = gt_seis[:, :, :, sample_slices_i].permute((0, 3, 1, 2, 4)).reshape(-1, C, T, X)
    gt_seis_slice_x = gt_seis[:, :, :, :, sample_slices_x].permute((0, 4, 1, 2, 3)).reshape(-1, C, T, I)
    gt_seis_slice_ix = torch.cat([gt_seis_slice_i, gt_seis_slice_x], dim=0)

    fake_seis_slice_t = fake_seis[:, :, sample_slices_t].permute((0, 2, 1, 3, 4)).reshape(-1, C, I, X)
    fake_seis_slice_i = fake_seis[:, :, :, sample_slices_i].permute((0, 3, 1, 2, 4)).reshape(-1, C, T, X)
    fake_seis_slice_x = fake_seis[:, :, :, :, sample_slices_x].permute((0, 4, 1, 2, 3)).reshape(-1, C, T, I)
    fake_seis_slice_ix = torch.cat([fake_seis_slice_i, fake_seis_slice_x], dim=0)

    return gt_seis_slice_t, gt_seis_slice_ix, fake_seis_slice_t, fake_seis_slice_ix


def main():
    G_model = G_Net(args.Gbase).cuda()
    D_model3d = D_Net3d(base=args.Dbase).cuda()
    D_model2d_ix = D_Net2d(base=args.Dbase).cuda()
    D_model2d_t = D_Net2d(base=args.Dbase).cuda()

    optm_G = torch.optim.Adam(G_model.parameters(), lr=args.G_lr, betas=args.Adam_beta)
    optm_D3D = torch.optim.Adam(D_model3d.parameters(), lr=args.D_lr, betas=args.Adam_beta)
    optm_D2Dix = torch.optim.Adam(D_model2d_ix.parameters(), lr=args.D_lr, betas=args.Adam_beta)
    optm_D2Dt = torch.optim.Adam(D_model2d_t.parameters(), lr=args.D_lr, betas=args.Adam_beta)

    advloss = AdversarialLoss(args.advfun, target_real_label=1.0, target_fake_label=0.0)

    dataset = seis_dataset(args)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.n_threads)
    scaler = GradScaler()

    G_model = torch.nn.DataParallel(G_model)
    D_model3d = torch.nn.DataParallel(D_model3d)
    D_model2d_ix = torch.nn.DataParallel(D_model2d_ix)
    D_model2d_t = torch.nn.DataParallel(D_model2d_t)


    if args.restore is not None:
        restore = torch.load(args.restore)
        G_model.load_state_dict(restore['G_model'])
        D_model3d.load_state_dict(restore['D_model3d'])
        D_model2d_ix.load_state_dict((restore['D_model2d_ix']))
        D_model2d_t.load_state_dict((restore['D_model2d_t']))

        optm_G.load_state_dict(restore['optm_G'])
        optm_D3D.load_state_dict(restore['optm_D3D'])
        optm_D2Dix.load_state_dict(restore['optm_D2Dix'])
        optm_D2Dt.load_state_dict(restore['optm_D2Dt'])


    loss_state = lossState(print_freq=args.print_freq)
    for idx, batch in enumerate(iterable=train_loader, start=1):
        gt_seis, masks = batch[0].cuda(), batch[1].cuda()  # mei you fan zhuan
        gt_seis = gt_seis * 2 - 1
        masked_seis = gt_seis * masks

        # =========================train GNet==============================
        with autocast():
            fake_seis = G_model(masked_seis)
            gt_seis_slice_t, gt_seis_slice_ix, fake_seis_slice_t, fake_seis_slice_ix \
                = get_2d_slice(gt_seis, fake_seis)

            g_fake3d = D_model3d(fake_seis)
            g_fake2d_ix = D_model2d_ix(fake_seis_slice_ix)
            g_fake2d_t = D_model2d_t(fake_seis_slice_t)

            g_gan_loss_3d = advloss(g_fake3d, True, False)
            g_gan_loss_2d = (advloss(g_fake2d_ix, True, False) + advloss(g_fake2d_t, True, False)) / 2
            tce_loss = TCE_loss()(fake_seis, gt_seis)
            #tce_loss = torch.nn.L1Loss()(fake_seis, gt_seis)
            g_gan_loss_3d, g_gan_loss_2d, tce_loss = g_gan_loss_3d.mean(), g_gan_loss_2d.mean(), tce_loss.mean()
            g_loss = g_gan_loss_3d*0.75 + g_gan_loss_2d*0.25 + tce_loss * 40

        optm_G.zero_grad()
        scaler.scale(g_loss).backward()
        scaler.step(optm_G)

        # ==========================train 3D DNet==========================
        with autocast():
            d_real3d = D_model3d(gt_seis.detach())
            d_fake3d = D_model3d(fake_seis.detach())
            d_loss_3d = (advloss(d_real3d, True, True) + advloss(d_fake3d, False, True)).mean() / 2
        optm_D3D.zero_grad()
        scaler.scale(d_loss_3d).backward()
        scaler.step(optm_D3D)

        # ==========================train 2D tline DNet=====================
        with autocast():
            d_real2d_t = D_model2d_t(gt_seis_slice_t)
            d_fake2d_t = D_model2d_t(fake_seis_slice_t.detach())
            d_loss_2d_t = (advloss(d_real2d_t, True, True) + advloss(d_fake2d_t, False, True)).mean() / 2
        optm_D2Dt.zero_grad()
        scaler.scale(d_loss_2d_t).backward()
        scaler.step(optm_D2Dt)

        # ===================train 2D iline and xline DNet===================
        with autocast():
            d_real2d_ix = D_model2d_ix(gt_seis_slice_ix)
            d_fake2d_ix = D_model2d_ix(fake_seis_slice_ix.detach())
            d_loss_2d_ix = (advloss(d_real2d_ix, True, True) + advloss(d_fake2d_ix, False, True)).mean() / 2
        optm_D2Dix.zero_grad()
        scaler.scale(d_loss_2d_ix).backward()
        scaler.step(optm_D2Dix)
        scaler.update()

        loss_state.update(g_gan_loss_3d.item(), d_loss_3d.item(), tce_loss.item(), g_gan_loss_2d.item(),
                          d_loss_2d_t.item(), d_loss_2d_ix.item(), idx)

        if idx % 100 == 0:
            comp_img = fake_seis * (1 - masks) + gt_seis * masks
            save_img((gt_seis + 1) / 2, (fake_seis + 1) / 2, (comp_img + 1) / 2, masks,
                     'output/' + str(idx).zfill(8) + '.png')

        if idx % 1000 == 0:
            state = {'G_model': G_model.state_dict(),
                     'D_model3d': D_model3d.state_dict(),
                     'D_model2d_ix': D_model2d_ix.state_dict(),
                     'D_model2d_t': D_model2d_t.state_dict(),
                     'optm_G': optm_G.state_dict(),
                     'optm_D3D': optm_D3D.state_dict(),
                     'optm_D2Dix': optm_D2Dix.state_dict(),
                     'optm_D2Dt': optm_D2Dt.state_dict()
                     }
            torch.save(state, '{:s}/MDA_{:d}.pth'.format('checkpoint', idx))


if __name__ == "__main__":
    main()
