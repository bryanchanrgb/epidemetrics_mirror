#!/bin/bash

pwd=$(pwd)
echo "Starting processing pipeline"

cd ${pwd}/src || exit
python3 ./main.py

echo "Pipeline processing finished"

echo "Running R scripts"
cd ${pwd}/R || exit
Rscript generate_figure_2.R
Rscript generate_figure_3.R
Rscript generate_figure_4.R

# Removing for now
# echo "Running web server"
# ln -s /data /web/data
# cd /web
# . /web/run-server.sh