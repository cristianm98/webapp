import base64
import os

import streamlit as st
import torch
import torchvision.models.segmentation
import torchvision.transforms as transforms

import transforms as ext_transforms
from arguments import get_sys_args
from models.pspnet import PspNet
from models.unet import UNet
from PIL import Image
from torchvision.transforms import ToPILImage


arguments = get_sys_args()
device = torch.device(arguments['device'])


def load_model(model, model_name, dataset_name):
    checkpoint_dir = arguments['checkpoint_dir']
    assert os.path.isdir(checkpoint_dir), \
        '\"{0}\" directory does not exist'.format(checkpoint_dir)
    checkpoint_path = get_checkpoint_path(checkpoint_dir, model_name, dataset_name)
    assert os.path.isfile(checkpoint_path), \
        '\"{0}\" file does not exist'.format(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    return model


def predict(model, image, class_encoding):
    image = transform_input(int(arguments['width']), int(arguments['height']))(image)
    original = ToPILImage()(image)
    image = torch.unsqueeze(image, 0)
    image = image.to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        if model.__class__.__name__.lower() == 'fcn':
            prediction = model(image)['out']
        else:
            prediction = model(image)
    _, prediction = torch.max(prediction.data, 1)
    prediction = prediction.detach().cpu()
    prediction = ext_transforms.LongTensorToRGBPIL(class_encoding)(prediction)
    result = merge_images(original, prediction)
    return result


def merge_images(image, seg_image):
    image = image.convert('RGBA')
    seg_image = seg_image.convert('RGBA')
    result = Image.blend(image, seg_image, 0.5)
    return result


def transform_input(width, height):
    image_transform = [transforms.Resize((width, height)), transforms.ToTensor()]
    return transforms.Compose(image_transform)


def get_model(model_name, num_classes, pretrained):
    if model_name == 'pspnet':
        return PspNet(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'unet':
        return UNet(num_classes=num_classes)
    elif model_name == 'fcn':
        return torchvision.models.segmentation.fcn_resnet101(num_classes=num_classes)


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


def get_checkpoint_path(checkpoint_dir, model_name, dataset_name):
    model_path = os.path.join(checkpoint_dir, dataset_name, model_name)
    return model_path
