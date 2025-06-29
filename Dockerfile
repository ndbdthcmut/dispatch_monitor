FROM python:3.12

RUN apt-get update && apt-get -y install \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libpython3-dev 

WORKDIR /au_yolo

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ./src .

CMD [ "python", "main.py" ]


