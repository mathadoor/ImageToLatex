FROM python:3.9

ARG NB_USER='matador'
RUN useradd -m $NB_USER

ENV HOME=/home/$NB_USER
ENV PATH="$HOME/.local/bin:${PATH}"

WORKDIR $HOME/img2latex

COPY . .

RUN chown -R $NB_USER:$NB_USER $HOME/img2latex

USER $NB_USER

RUN /usr/local/bin/python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
