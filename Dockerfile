FROM tensorflow/tensorflow:1.14.0-py3

USER root

RUN \
	apt-get update && apt-get install -q -y \
		libsm6 libxext6 libxrender-dev \
		python3-pip &&\
	pip3 install opencv-python scipy

RUN apt-get install -qqy x11-apps
ENV DISPLAY :0
ENV QT_X11_NO_MITSHM=1
CMD xeyes

WORKDIR /home/env
