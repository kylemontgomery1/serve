cd /scratch/serve/model_store
echo "Converting Model"
torch-model-archiver --model-name llama-2-70b-chat --version 1.0 --handler /scratch/serve/custom_handler/text_generation_handler_multigpu.py --extra-files llama-2-70b-chat.zip -r requirements.txt -f -c llama-2-70b-chat-config.yaml