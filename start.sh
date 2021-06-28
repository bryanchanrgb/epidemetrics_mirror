#!/bin/bash

echo "Starting processing pipeline"

cd /src
python3 ./main.py

echo "Pipeline processing finished"

echo "Running web server"
ln -s /data /web/data
cd /web
. /web/run-server.sh