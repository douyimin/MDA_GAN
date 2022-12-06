import torch.nn as nn
import torch


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class D_Net3d(nn.Module):
    def __init__(self, in_channels=1, base=72, use_spectral_norm=True):
        super(D_Net3d, self).__init__()

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels=in_channels, out_channels=base, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels=base, out_channels=base * 2, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels=base * 2, out_channels=base * 4, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels=base * 4, out_channels=base * 8, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels=base * 8, out_channels=1, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        outputs = conv5
        return outputs


class D_Net2d(nn.Module):
    def __init__(self, in_channels=1, base=72, use_spectral_norm=True):
        super(D_Net2d, self).__init__()

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=base, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=base, out_channels=base * 2, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=base * 2, out_channels=base * 4, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=base * 4, out_channels=base * 8, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=base * 8, out_channels=1, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        outputs = conv5
        return outputs


class AdversarialLoss(nn.Module):
    """
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        """
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCEWithLogitsLoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss(reduction='mean')

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs).cuda()
            loss = self.criterion(outputs, labels)
            return loss


class TCE_loss(nn.Module):
    def __init__(self):
        super(TCE_loss, self).__init__()

    def forward(self, y, target):
        y = torch.where(y == 1, 1 - 1e-7, y)
        y = torch.where(y == -1, -1 + 1e-7, y)

        loss = ((1 + target) / 2) * torch.log((1 + y) / 2) + ((1 - target) / 2) * torch.log((1 - y) / 2)

        with torch.no_grad():
            mask = 1 - torch.isinf(loss).float()
            mask = ((1 - torch.isnan(loss).float()) == mask)

        loss = -torch.sum(torch.where(mask, loss, 0)) / torch.sum(mask)

        return loss
