import sys
from argparse import ArgumentTypeError
from argparse import ArgumentParser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't'):
        return True
    elif v.lower() in ('false', 'f'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        '-d',
        default='cuda',
        help='Device on which the network will be trained. Default: cuda')
    parser.add_argument(
        '-w',
        type=int,
        default=512,
        help='The width of the image. Default: 512'),
    parser.add_argument(
        '-h',
        type=int,
        default=512,
        help='The width of the image. Default: 512'),
    parser.add_argument(
        '-c',
        type=str,
        default='/content/drive/MyDrive/Colab Notebooks/Checkpoints',
        help='Path to the saved checkpoints. Default: /content/drive/MyDrive/Colab Notebooks/Checkpoints')
    return parser.parse_args()


def get_sys_args():
    return {
        'device': sys.argv[1],
        'checkpoint_dir': sys.argv[2],
        'width': sys.argv[3],
        'height': sys.argv[4],
        'use_day': str2bool(sys.argv[5])
    }
