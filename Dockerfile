FROM python:3.9.5-slim

WORKDIR /src

RUN pip install --upgrade pip setuptools

COPY requirements.txt .
RUN pip3 install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY src /src
COPY web /web

COPY start.sh /start.sh

# command to run on container start
CMD [ "/bin/sh", "/start.sh" ]