# style.py
import torch, os
import streamlit as st
import torchvision.transforms as transforms
from PIL import Image
import sys
sys.path.append("..")

from train.models import VanillaWAP
from train.utils.global_params import BASE_CONFIG, VOCAB_LOC
from train.utils.datasets import convert_to_string, get_vocabulary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model():
    with torch.no_grad():
        model = VanillaWAP(BASE_CONFIG)
        model_loc = os.path.join(BASE_CONFIG['root_loc'], BASE_CONFIG['train_params']['save_loc'])
        state_dict = torch.load('checkpoints/model_30.pth', map_location=device)

        model.load_state_dict(state_dict)
        model.eval().to(device)
        return model


@st.cache_resource
def translate(_model, content_image):
    img = Image.open(content_image).convert('L')
    transform = transforms.Compose([transforms.ToTensor()])
    vocab = get_vocabulary(VOCAB_LOC)
    index_to_word = {i: word for i, word in enumerate(vocab)}
    img = transform(img).unsqueeze(0).to(device)
    if torch.mean(img) > 0.5:
        img = 1 - img   # invert the image
    mask = torch.ones_like(img).to(device)

    with torch.no_grad():
        tokenized_label, alphas = _model.translate(img, mask=mask)

    label = convert_to_string(tokenized_label, index_to_word)
    return label, alphas
