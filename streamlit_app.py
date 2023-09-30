import base64
import streamlit as st
from st_click_detector import click_detector
from PIL import Image
import numpy as np
from translator import inference
import matplotlib.pyplot as plt

# The following functions are copied from https://github.com/vivien000/st-click-detector/issues/4 to display local
# images

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

numpy_array = None
if input_image is not None:
    image = Image.open(input_image)
    numpy_array = np.array(image)
    # Plot the numpy image to matplotlib figure in grayscale
    fig, ax = plt.subplots()
    ax.imshow(numpy_array, cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

clicked = st.button('Translate Text')

labels_clicked = []
alphas = None
if clicked and input_image is not None:
    model = inference.load_model()
    label, alphas = inference.translate(model, input_image)
    st.write('### Translated Latex Encoding:')
    st.write(label)
    # Show the tokens as buttons
    # label_show = label.split(' ')
    # max_val = 100
    # a, b = 0.98, 6
    # num_labels = len(label_show)
    # curr_l = 0
    # columns_array = []
    # start = 0
    # for i in range(num_labels):
    #     l = label_show[i]
    #     new_l = a * len(l) + b
    #     if curr_l + new_l <= max_val and i != num_labels - 1:
    #         columns_array += [new_l]
    #         curr_l += new_l
    #     elif len(columns_array) != 0:
    #         columns_array = columns_array if curr_l == max_val else columns_array + [max_val - curr_l]
    #         columns = st.columns(columns_array)
    #         for j in range(start, i):
    #             with columns[j - start]:
    #                 labels_clicked.append(st.button(label_show[j], key=j, use_container_width=True))
    #
    #         start = i
    #         if curr_l + new_l > max_val:
    #             columns_array = [new_l]
    #             curr_l = new_l
    #         else:
    #             columns_array = []
    #             curr_l = 0

    input_image = None
elif clicked:
    st.write('Please upload an image or select one from the above thumbnails.')


