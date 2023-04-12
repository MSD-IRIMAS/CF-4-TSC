FROM tensorflow/tensorflow:latest-gpu
RUN
RUN apt update
RUN pip install numpy pandas scikit-learn matplotlib