import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--Gbase', default=48, type=int, help='GNet base width')
parser.add_argument('--Dbase', default=64, type=int, help='DNet base width')
parser.add_argument('--G_lr', default=0.0001, type=float, help='GNet lr')
parser.add_argument('--D_lr', default=0.0004, type=float, help='DNet lr')
parser.add_argument('--Adam_beta', default=(0., 0.9), type=tuple, help='')
parser.add_argument('--advfun', default='lsgan', type=str, help='type = nsgan | lsgan | hinge')
parser.add_argument('--sample_slices', default=8, type=int, help='sample slice from tline iline xline')
parser.add_argument('--n_threads', type=int, default=10,help='')
parser.add_argument('--batch_size', type=int, default=4, help='batch size of trainset')
parser.add_argument('--train_path', default=r'/home/emin/GAN_TRAIN', help='fixed trainset root path')
parser.add_argument('--restore', type=str, default=None, help='pretrain model')
parser.add_argument('--data_shape', type=tuple, default=(128,128,128), help='train data shape')
parser.add_argument('--train_step', type=int, default=500000, help='')

parser.add_argument('--ema_model', type=bool, default=True, help='not imp')




args = parser.parse_args()
