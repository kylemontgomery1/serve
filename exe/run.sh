#!/bin/bash

bsub -J "Inference API" -m chenguang01 -gpu "num=8:gmem=48GB" /bin/bash exe/launch_docker.sh