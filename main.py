import streamlit as st
from PIL import Image

import color_encoding
import utils

st.set_option('deprecation.showfileUploaderEncoding', False)
utils.set_png_as_page_bg('background.jpg')
st.title("Road Detection - Semantic Segmentation")
st.write("")
model_options = ('PSPNet', 'UNet', 'FCN')
dataset_options = ('CamVid', 'InfraRed')
selected_dataset_idx = st.selectbox('Dataset used for training', list(range(len(dataset_options))),
                                    format_func=lambda x: dataset_options[x])
dataset_name = dataset_options[selected_dataset_idx]
st.write("Selected dataset: {0}".format(dataset_name))
selected_model_idx = st.selectbox('Model', list(range(len(model_options))),
                                  format_func=lambda x: model_options[x])
model_name = model_options[selected_model_idx]
st.write("Selected model: {0}".format(model_name))
class_encoding = color_encoding.get_color_encoding(dataset_name)
file_up = st.file_uploader("Upload an image")
if file_up is not None:
    model = utils.get_model(model_name, len(class_encoding), True)
    model = utils.load_model(model, model_name.lower(), dataset_name.lower())
    input_image = Image.open(file_up)
    st.image(input_image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Just a second...")
    output_image = utils.predict(model, input_image, class_encoding)
    st.image(output_image, caption="Result", use_column_width=True)
