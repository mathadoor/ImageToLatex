# Function to validate the latex expressions in the dataset
import os, re
from pylatexenc.latexencode import utf8tolatex
from data_utils import get_path

CROHME_TRAIN = get_path(kind = 'train')

def validate_latex(latex_str):
    try:
        utf8tolatex(latex_str)
        return True
    except:
        return False

# Use a valid LaTeX string
latex_str = r'\frac{1}{2}'
print(validate_latex(latex_str))  # Outputs: True

# Use an invalid LaTeX string
latex_str = r'\frac{1}{'
print(validate_latex(latex_str))  # Outputs: False
