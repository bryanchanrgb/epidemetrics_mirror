#!/bin/bash

echo "Starting processing pipeline"

python3 ./main.py
#ls -las

echo "Pipeline processing finished"

ln -s /data /web/data
cd /web
ls -las /web/
ls -las /web/data/
. /web/run-server.sh