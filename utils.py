import os

import torch
import torchvision.transforms as transforms
import transforms as ext_transforms
import time
from arguments import get_arguments

arguments = get_arguments()
load_dir_path = arguments.checkpoint_dir
device = torch.device(arguments.device)


def load_model(model, model_name, dataset_name):
    checkpoint_path = os.path.join(load_dir_path, 'best_' + model_name + "_" + dataset_name)
    assert os.path.isdir(load_dir_path), \
        '\"{0}\" directory does not exist'.format(load_dir_path)
    assert os.path.isfile(checkpoint_path), \
        '\"{0}\" file does not exist'.format(checkpoint_path)
    start_time = time.time()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print("[Load checkpoint] --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    model.load_state_dict(checkpoint['model'])
    print("[Load model] --- %s seconds ---" % (time.time() - start_time))
    return model


def predict(model, image, class_encoding):
    start_time = time.time()
    image = transform_input(512, 512)(image)
    print("[Transform input] --- %s seconds ---" % (time.time() - start_time))
    image = torch.unsqueeze(image, 0)
    image = image.to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        prediction = model(image)
        print("[Predict]--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    _, prediction = torch.max(prediction.data, 1)
    print("[Max]--- %s seconds ---" % (time.time() - start_time))
    # tf = transforms.Compose([
    #     ext_transforms.LongTensorToRGBPIL(class_encoding),
    # ])
    start_time = time.time()
    prediction = ext_transforms.LongTensorToRGBPIL(class_encoding)(prediction)
    print("[Ext transform]--- %s seconds ---" % (time.time() - start_time))
    return prediction


def transform_input(width, height):
    image_transform = [transforms.Resize((width, height)), transforms.ToTensor()]
    return transforms.Compose(image_transform)
