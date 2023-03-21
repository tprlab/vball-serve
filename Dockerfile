FROM python:3.8-bullseye

RUN apt update

RUN pip3 install numpy
RUN pip3 install scikit-learn
RUN pip3 install lazypredict

ADD predict.py /
ADD trdata.json /

CMD ["python3", "predict.py"]