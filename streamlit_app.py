import base64
import streamlit as st
from st_click_detector import click_detector
from PIL import Image
import numpy as np
from translator import inference
import matplotlib.pyplot as plt

st.set_page_config(page_title='Img2LATeX', page_icon=':pencil2:')


# The following functions are copied from https://github.com/vivien000/st-click-detector/issues/4 to display local
# images
def initialize_state():
    st.session_state['initialized'] = True
    st.session_state['selected_image'] = None
    st.session_state['np_image'] = None
    st.session_state['label'] = None
    st.session_state['alphas'] = None
    st.session_state['active_alpha'] = None
    st.session_state['type_input'] = 'From a Pre-Existing Set'


def reset_state(key, value):
    initialize_state()
    st.session_state[key] = value


def base64img(path: str):
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
        return data


def images_html(examples):
    contents = [
        f"<a href='#' id='{i}'><img width='180' alt='{examples[i]}' src='data:image/png;base64,{base64img(path)}'></a>"
        for i, path in enumerate(examples)]
    return f'{"&nbsp;" * 2}'.join(contents)


def active_alpha(index):
    st.session_state['active_alpha'] = index


if 'initialized' not in st.session_state:
    initialize_state()

image_arrays = [
    "data/CROHME/train/off_image_train/70_carlos_0.bmp",
    "data/CROHME/train/off_image_train/93_alfonso_0.bmp",
    "data/CROHME/train/off_image_train/94_bruno_0.bmp",
]

st.title('Handwritten Equations to Latex Translator')

st.write('### Introduction:')
st.write('''This is a demo of a model that translates handwritten equations to latex. The model was trained on the 
[CROHME](https://researchdata.edu.au/crohme-competition-recognition-expressions-png/639782) dataset. The model implements
the watch-attend-parse architecture from [this paper](https://arxiv.org/abs/1707.01294). The details of the project
can be found in its [repository](https://github.com/mathadoor/ImageToLatex). 

The application on this page is a demo of the model. It provides two ways to the user to select an image to translate.
The first is to select an image from a pre-existing set of images. The second is to upload an image. The user can then
click on the translate button to translate the image. Subsequently, the user can toggle the attention map, and click on
the tokens to see which parts of the image the model is attending to. 

Note: The model is not perfect and can make mistakes. Also, it is not trained on all possible symbols and can make
 mistakes on the ones it has not seen''')

st.write('### Input image:')
st.session_state['type_input'] = st.selectbox('How would you like to input the image?',
                                              ('From a Pre-Existing Set', 'Upload Image'))

if st.session_state['type_input'] == 'Upload Image':
    reset_state('type_input', 'Upload Image')
    st.session_state['selected_image'] = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg', 'bmp'])
else:
    select_image = click_detector(images_html(image_arrays))
    if select_image == "":
        select_image = "0"
    st.session_state['selected_image'] = image_arrays[int(select_image)]


if st.session_state['selected_image'] is not None:
    st.write('### Selected Image:')
    image = Image.open(st.session_state['selected_image'])
    numpy_array = np.array(image) / 255.0
    # Plot the numpy image to matplotlib figure in grayscale
    fig, ax = plt.subplots()
    ax.imshow(numpy_array)

    if st.session_state['active_alpha'] is not None:
        index = st.session_state['active_alpha']
        expected_shape = list(numpy_array.shape)
        attention = inference.pass_attention(st.session_state['alphas'][index], expected_shape)
        st.session_state['active_alpha'] = None
        ax.imshow(attention, cmap='gray', alpha=0.4, extent=(0, expected_shape[1], expected_shape[0], 0))

    ax.axis('off')
    st.pyplot(fig)

clicked = st.button('Translate Text')

if clicked and st.session_state['selected_image'] is not None:
    model = inference.load_model()
    label, alphas = inference.translate(model, st.session_state['selected_image'])
    st.session_state['label'] = label
    st.session_state['alphas'] = alphas

attention_show = st.toggle('Show Attention Map', value=False)
if st.session_state['label'] is not None:
    st.write('### Translated Latex Encoding:')
    if not attention_show:
        st.write(st.session_state['label'])
    else:
        st.caption('Click on a token to see the attention map i.e what corresponding image patches are being attended '
                   'to')
        # Show the tokens as buttons
        label_show = st.session_state['label'].split(' ')
        max_val = 100
        a, b = 0.98, 6
        num_labels = len(label_show)
        curr_l = 0
        columns_array = []
        start = 0
        for i in range(num_labels):
            l = label_show[i]
            new_l = a * len(l) + b
            if curr_l + new_l <= max_val and i != num_labels - 1:
                columns_array += [new_l]
                curr_l += new_l
            elif len(columns_array) != 0:
                columns_array = columns_array if curr_l == max_val else columns_array + [max_val - curr_l]
                columns = st.columns(columns_array)
                for j in range(start, i):
                    with columns[j - start]:
                        st.button(label_show[j], key=j, on_click=active_alpha, kwargs={'index': j},
                                  use_container_width=True)
                start = i
                if curr_l + new_l > max_val:
                    columns_array = [new_l]
                    curr_l = new_l
                else:
                    columns_array = []
                    curr_l = 0

        st.button('Reset Attention', key=-1, on_click=active_alpha, kwargs={'index': None}, use_container_width=True)
