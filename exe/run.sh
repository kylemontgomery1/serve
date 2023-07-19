#!/bin/bash

bsub -J "Inference API" -m chenguang02 -gpu "num=4:gmodel=NVIDIARTXA6000:mode=exclusive_process" /bin/bash /scratch/serve/exe/launch_docker.sh
