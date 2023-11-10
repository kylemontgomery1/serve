#!/bin/bash

export TEMP=/scratch/tmp
torchserve --start --ncs --ts-config model_store/config.properties