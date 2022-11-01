FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04
# :10.2-cudnn7-devel-ubuntu18.04
RUN apt update
#RUN apt install software-properties-common  \
#RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get install -y python3-pip
#FROM python:3.9
WORKDIR /code
# WORKDIR /app
COPY ./requirements2.txt /code/requirements.txt
# COPY requirements2.txt ./
# CMD ["chmod +x check_init_model.sh"]
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
# RUN pip install --no-cache-dir -r requirements2.txt
# COPY ./Codes /code/train
# WORKDIR /code/train
# CMD ["nohup", "python3", "Model_train.py", "> log_error.txt"]
COPY ./app /code/app
EXPOSE 8002
WORKDIR /code/app
# CMD ["./check_init_model.sh"]
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "80", "--log-level", "error"]
