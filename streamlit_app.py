import base64
import streamlit as st
from st_click_detector import click_detector
from PIL import Image
import numpy as np
from translator import inference
import matplotlib.pyplot as plt


# The following functions are copied from https://github.com/vivien000/st-click-detector/issues/4 to display local images
def base64img(path: str):
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
        return data


def images_html(examples):
    contents = [
        f"<a href='#' id='{i}'><img width='180' alt='{examples[i]}' src='data:image/png;base64,{base64img(path)}'></a>"
        for i, path in enumerate(examples)]
    return f'{"&nbsp;" * 2}'.join(contents)


st.title('Handwritten Equations to Latex Translator')
image_arrays = [
    "data/CROHME/train/off_image_train/70_carlos_0.bmp",
    "data/CROHME/train/off_image_train/93_alfonso_0.bmp",
    "data/CROHME/train/off_image_train/94_bruno_0.bmp",
]

st.write('### Input image:')
option = st.selectbox('How would you like to input the image?', ('From a Pre-Existing Set', 'Upload Image'))
input_image = None
if option == 'Upload Image':
    input_image = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg', 'bmp'])
    # if input_image is not None:
    #     st.image(input_image, width=700)  # image: numpy array
else:
    select_image = click_detector(images_html(image_arrays))
    if select_image != "":
        input_image = image_arrays[int(select_image)]
        st.write('#### Selected image:')
        # st.image(input_image, width=700)  # image: numpy array
    else:
        st.write('Please select an image from the above thumbnails.')

if input_image is not None:
    image = Image.open(input_image)
    numpy_array = np.array(image)
    # Plot the numpy image to matplotlib figure in grayscale
    fig, ax = plt.subplots()
    ax.imshow(numpy_array, cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

clicked = st.button('Translate Text')

if clicked and input_image is not None:
    model = inference.load_model()
    label, alphas = inference.translate(model, input_image)
    st.write('### Translated Latex Encoding:')
    st.write(label)
    input_image = None
elif clicked:
    st.write('Please upload an image or select one from the above thumbnails.')
