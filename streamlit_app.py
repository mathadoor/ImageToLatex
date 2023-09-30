import streamlit as st
from PIL import Image
from translator import inference

st.title('Handwritten Equations to Latex Translator')

# img = st.sidebar.selectbox(
#     'Select Image',
#     ('amber.jpg', 'cat.png')
# )
#
# style_name = st.sidebar.selectbox(
#     'Select Style',
#     ('candy', 'mosaic', 'rain_princess', 'udnie')
# )


input_image = "data/CROHME/train/off_image_train/70_carlos_0.bmp"

st.write('### Input image:')
image = Image.open(input_image)
st.image(input_image, width=400) # image: numpy array

clicked = st.button('Translated Text')

if clicked:
    model = inference.load_model()
    label = inference.translate(model, input_image)
    st.write('### Translated image:')
