import streamlit as st
import torch
from PIL import Image

import color_encoding
import utils
from arguments import get_sys_args

arguments = get_sys_args()
device = torch.device(arguments['device'])

st.set_option('deprecation.showfileUploaderEncoding', False)
utils.set_png_as_page_bg('background.jpg')
st.title("Road Detection - Semantic Segmentation")
st.write("")
model_options = ('PSPNet', 'UNet', 'FCN')
dataset_options = ('InfraRed', 'CamVid')
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
    model = utils.get_model(model_name.lower(), len(class_encoding), True)
    model = utils.load_model(model, model_name.lower(), dataset_name.lower())
    input_image = Image.open(file_up)
    st.image(input_image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    submit_btn = st.button('Submit')
    if submit_btn:
        st.write("Just a second...")
        if device.type == 'cuda':
            output_image, inference_time = utils.predict_cuda(model, input_image, class_encoding)
            inference_time = inference_time / 1000
        else:
            output_image, inference_time = utils.predict_cpu(model, input_image, class_encoding)
        st.image(output_image, caption="Result\nInference time: {0:.3f} seconds".format(inference_time), use_column_width=True)
