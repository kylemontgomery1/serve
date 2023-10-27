# AgentInstruct Inference API

### Setup
The TorchServe API is best run within their official docker container found [here](https://hub.docker.com/r/pytorch/torchserve). To install the necessary packages, run:
```
python ./ts_scripts/install_dependencies.py --cuda=cu117
```
where cu117 correspond to CUDA 11.7. Adjust as needed. Additional model-specific packages should be put in model_store/requirements.txt, and will be installed when a model is assigned workers. 

Then adjust the ```model_store``` path in ```serve/model_store/config.properties```. 

### To boot up the api
```
cd model_store
export TEMP=/tmp
torchserve --start --ncs --ts-config config.properties

```
This will load the api, but will not register any models or load any workers.

### Handler and model config
Each model should have an associated handler file and a model configuration. 
A handler file contains 4 key functions:

    1. initialize (initalize workers)
    2. preprocess (preprocess request)
    3. inference (executre request)
    4. postprocess (return request)
    
Our handling code was adapted from [TogetherAI](https://github.com/togethercomputer/Quick_Deployment_HELM) to insure continuity between our early experimentation and main results. Handler's can contain model-specific code. For example, out Llama-2-chat handlers adapt the raw request into Meta's chat format, and our Llama-2-70b and Llama-2-70b-chat handlers contain options to better map workers to GPUs, as well as various quantization options.

An example model config for Llama-2-70b-chat is as follows:
```
responseTimeout: 5000
torchrun:
    nproc-per-node: 1 
handler:
    model_path: "/path/to/model/checkpoint"
    quantize: "nf4"
    num_gpu_per_model: 1
    gpu_max_memory: 4.8e10 # note we assume a homogeneous gpu setup.
```
Variables set in the model config can be directly accessed by the handler. For example `ctx.model_yaml_config["handler"]["model_path"]` will grab the model path specified in the config.

To generate a runtime file for a model, run
```
torch-model-archiver --model-name [model_name] --version 1.0 --handler /scratch/serve/custom_handler/[model_name]-handler.py  -r requirements.txt -f -c [model_name]-config.yaml --archive-format tgz
```

### To register a model with an initial number of workers
```
curl -X POST "http://localhost:8081/models?url={model_name}.tar.gz&initial_workers={num_workers}"
```

### To check the status of a model
```
curl http://localhost:8081/models/{model_name}
```

### To send an inference request to a model
Inference requests will be sent over localhost:8080, and api management is through localhost:8081.
```
endpoint = 'http://0.0.0.0:8080/predictions/{model_name}'
response = requests.post(endpoint, json={
    "seed": 0,
    "prompt": "Alan Turing was a ",
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 50,
    "logprobs": 0,
    "stop": [],
})
```
Note that the API is not set up for batching, so `prompt` should be a string.

### To stop the api
```
torchserve --stop
```
