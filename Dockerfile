FROM python:3.9.5-slim

WORKDIR /

RUN pip install --upgrade pip setuptools

COPY requirements.txt .
RUN pip3 install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY src /src
COPY web /web
COPY tests /tests

COPY start.sh /start.sh
COPY run_tests.sh /run_tests.sh

# command to run on container start
CMD [ "/bin/sh", "/start.sh" ]