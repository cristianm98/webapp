from collections import OrderedDict

import streamlit as st
from PIL import Image

import utils
from models.pspnet import PspNet

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Road Detection - Semantic Segmentation")
st.write("")

file_up = st.file_uploader("Upload an image")

# TODO complete these
model_name = "PSPNet"
# model_name = "UNet"
dataset_name = "camvid"
class_encoding = OrderedDict([
    ('Animal', (64, 128, 64)),
    ('Archway', (192, 0, 128)),
    ('Bicyclist', (0, 128, 92)),
    ('Bridge', (0, 128, 64)),
    ('Building', (128, 0, 0)),
    ('Car', (64, 0, 128)),
    ('CartLuggagePram', (64, 0, 192)),
    ('Child', (192, 128, 64)),
    ('Column_Pole', (192, 192, 128)),
    ('Fence', (64, 64, 128)),
    ('LaneMkgsDriv', (128, 0, 192)),
    ('LaneMkgsNonDriv', (192, 0, 64)),
    ('Misc_Text', (128, 128, 64)),
    ('MotorcycleScooter', (192, 0, 192)),
    ('OtherMoving', (128, 64, 64)),
    ('ParkingBlock', (64, 192, 128)),
    ('Pedestrian', (64, 64, 0)),
    ('Road', (128, 64, 128)),
    ('RoadShoulder', (128, 128, 192)),
    ('Sidewalk', (0, 0, 192)),
    ('SignSymbol', (192, 128, 128)),
    ('Sky', (128, 128, 128)),
    ('SUVPickupTruck', (64, 128, 192)),
    ('TrafficCone', (0, 0, 64)),
    ('TrafficLight', (0, 64, 64)),
    ('Train', (192, 64, 128)),
    ('Tree', (128, 128, 0)),
    ('Truck_Bus', (192, 128, 192)),
    ('Tunnel', (64, 0, 64)),
    ('VegetationMisc', (192, 192, 0)),
    ('Void', (0, 0, 0)),
    ('Wall', (64, 192, 0))
])

if file_up is not None:
    model = PspNet(len(class_encoding), pretrained=True)
    # model = Unet(len(class_encoding))
    model = utils.load_model(model, model_name, dataset_name)
    input_image = Image.open(file_up)
    st.image(input_image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Just a second...")
    output_image = utils.predict(model, input_image, class_encoding)
    st.image(output_image, caption="Result", use_column_width=True)
