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
        '--device',
        default='cuda',
        help='Device on which the network will be trained. Default: cuda')
    parser.add_argument(
        '--width',
        type=int,
        default=512,
        help='The width of the image. Default: 512'),
    parser.add_argument(
        '--height',
        type=int,
        default=512,
        help='The width of the image. Default: 512'),
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='/content/drive/MyDrive/Colab Notebooks/Checkpoints',
        help='Path to the saved checkpoints. Default: /content/drive/MyDrive/Colab Notebooks/Checkpoints')
    parser.add_argument(
        '--model',
        choices=['pspnet', 'unet'],
        default='unet',
        help='The model to use. Default: unet')
    return parser.parse_args()
