FROM python:3.10

ARG NB_USER='matador'
RUN useradd -m $NB_USER

ENV HOME=/home/$NB_USER
ENV PATH="$HOME/.local/bin:${PATH}"

WORKDIR $HOME/ImageToLatex

# Install torch and torchvision first
RUN /usr/local/bin/python -m pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchtext torcheval

COPY requirements.txt .
RUN /usr/local/bin/python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R $NB_USER:$NB_USER $HOME/ImageToLatex

USER $NB_USER
