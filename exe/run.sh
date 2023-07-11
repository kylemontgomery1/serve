#!/bin/bash

bsub -J Inference API -m chenguang02 -gpu "num=8:gmodel=NVIDIARTXA6000:mode=exclusive_process" /bin/bash /scratch/serve/exe/launch_docker.sh