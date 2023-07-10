#!/bin/bash

cd /scratch/serve/model_store
export TEMP=/scratch/tmp

torchserve --start --ncs --ts-config config.properties
