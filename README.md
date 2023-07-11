# Chenguang02 Text Generation Inference API

### To load api
```
bash /scratch/serve/exe/run.sh
```
This will load the api, but will not register any models or load any workers. It requires all the GPU's in the node to be empty, since it will run them in exclusive process mode.

### To register a model with an initial number of workers
```
curl -X POST "http://localhost:8081/models?url={model_name}.mar&initial_workers={num_workers}"
```

### To check the status of a model
```
curl http://localhost:8081/models/{model_name}
```

### To send an inference request to a model
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

### To deploy additional models
1. Launch singularity shell
   ```
   singularity run --nv /scratch/serve.sif /bin/bash
   ```
2. Download weights from HuggingFace to `/scratch/serve/model_store/{model_name}`
3. Zip the model weights
   ```
   cd /scratch/serve/model_store/{model_name}
   zip -r /scratch/serve/model_store/{model_name}.zip *
   ```
4. Create `.mar` runtime file
   ```
   cd /scratch/serve/model_store
   torch-model-archiver --model-name {model_name} --version 1.0 --handler /scratch/serve/custom_handler/text_generation_handler.py --extra-files {model_name}.zip -r requirements.txt -f
   ```  
Note that the handler assumes that each model can fit on a single GPU. Also, if you need any additional packages to run the model, add them to the /scratch/serve/model_store/requirements.txt before creating the `.mar` file.

### To stop the api
```
singularity run --nv /scratch/serve.sif /bin/bash
export TEMP=/scratch/tmp
torchserve --stop
```

## Beware the following sections. Worker reallocation is a bit tricky, as the workers often crash or try to run on gpus alread in use. You're much better off if you can assign workers when you register the model.

### To add/increase workers for a model
```
curl -v -X PUT "http://localhost:8081/models/{model_name}?min_worker={num_workers}"
```

### To remove/decrease workers for a model

```
curl -v -X PUT "http://localhost:8081/models/{model_name}?min_worker={num_workers}"
```
Be careful not to assign more than 8 workers in total between all models, otherwise models may start to OOM. So always remove workers before adding workers.
