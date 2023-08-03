#!/bin/bash

bsub -J "Inference API" -m chenguang02 -gpu "num=8:gmodel=NVIDIARTXA6000:gmem=48" /bin/bash /scratch/serve/exe/launch_docker.sh
