import ast
import json
import logging
import os
from abc import ABC
import numpy as np
import random
import timeit
import zipfile

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)


class TransformersSeqClassifierHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(TransformersSeqClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the BERT model is loaded and
        the Layer Integrated Gradients Algorithm for Captum Explanations
        is initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        
        self.task_info = {
            "seed": 0,
            "prompt_seqs": None,
            "output_len": 16,
            "beam_width": 1,
            "top_k": 50,
            "top_p": 0,
            "beam_search_diversity_rate": 0,
            "temperature": 0.1,
            "len_penalty": 0,
            "repetition_penalty": 1.0,
            "stop": [],
            "logprobs": 0,
        }
        
        with zipfile.ZipFile(model_dir + "/model.zip", "r") as zip_ref:
            zip_ref.extractall(model_dir + "/model")
        
        self.model = AutoModelForCausalLM.from_pretrained(model_dir + "/model")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir + "/model")

        self.model.to(self.device)
        self.model.eval()
        
        torch.manual_seed(0)
        torch.cuda.empty_cache()
        
        logger.info("Transformer model from path %s loaded successfully", model_dir)

        self.initialized = True

    def preprocess(self, requests):
        """Basic text preprocessing, based on the user's chocie of application mode.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of Tensor for the size of the word tokens.
        """
        requests = {k: v for k, v in requests.items() if v is not None}
        
        self.task_info["seed"] = get_int(requests.get("seed", 0), default=0)
        self.task_info["prompt_seqs"] = [str(requests['prompt'])]
        self.task_info["output_len"] = get_int(requests.get("max_tokens", 16), default=16)
        self.task_info["beam_width"] = get_int(requests.get("beam_width", 1), default=1)
        self.task_info["top_k"] = get_int(requests.get("top_k", 50), default=50)
        self.task_info["top_p"] = get_float(requests.get("top_p", 0.0), default=0.0)
        self.task_info["beam_search_diversity_rate"] = get_float(requests.get("beam_search_diversity_rate", 0.0),
                                                                 default=0.0)
        self.task_info["temperature"] = get_float(requests.get("temperature", 0.8), default=0.8)
        self.task_info["len_penalty"] = get_float(requests.get("len_penalty", 0.0), default=0.0)
        self.task_info["repetition_penalty"] = get_float(requests.get("repetition_penalty", 1.0), default=1.0)
        self.task_info["stop"] = requests.get("stop", [])
        self.task_info["logprobs"] = get_int(requests.get("logprobs", 0), default=0)
        
        return self.tokenizer(self.task_info["prompt_seqs"], return_tensors="pt", device = self.device)


    def inference(self, inputs):
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        Args:
            input_ids: Text Tensor from the pre-process function is passed here
        Returns:
            list : It returns the predicted value for the input text
        """
        
        with torch.no_grad():
            torch.manual_seed(self.task_info['seed'])
            np.random.seed(self.task_info['seed'])
            random.seed(self.task_info['seed'])
            
            input_length = inputs.input_ids.shape[1]
            output_scores = self.task_info["logprobs"] > 0
            
            time = timeit.default_timer()
            if self.task_info["temperature"] == 0:
                outputs = self.model.generate(
                    **inputs, do_sample=True, top_p=self.task_info['top_p'],
                    temperature=1.0, top_k=1,
                    max_new_tokens=self.task_info["output_len"],
                    return_dict_in_generate=True,
                    output_scores=output_scores,  # return logit score
                    output_hidden_states=False,  # return embeddings
                )
            else:
                outputs = self.model.generate(
                    **inputs, do_sample=True, top_p=self.task_info['top_p'],
                    temperature=self.task_info["temperature"],
                    max_new_tokens=self.task_info["output_len"],
                    return_dict_in_generate=True,
                    output_scores=output_scores,  # return logit score
                    output_hidden_states=False,  # return embeddings
                )
            if output_scores:
                self.logprobs = convert_hf_score_to_logprobs(outputs.scores, self.task_info["logprobs"], self.tokenizer)
            else:
                self.logprobs = None

            self.time_elapsed = timeit.default_timer() - time
            
        inference_result = []
        item = {'choices': [], }
        for beam_id in range(self.task_info["beam_width"]):
            token = outputs.sequences[beam_id, input_length:]
            output = self.tokenizer.decode(token)
            choice = {
                "text": post_processing_text(output, self.task_info["stop"]),
                "index": beam_id,
                "finish_reason": "length"
            }
            item['choices'].append(choice)
            inference_result.append(item)
        
        return inference_result

    def postprocess(self, inference_results):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_results (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        
        result = {
            "choices": inference_results[0]['choices'],
            "raw_compute_time": self.time_elapsed,
        }
        
        if self.task_info["logprobs"] > 0:
            result['logprobs'] = self.logprobs
        
        return result
    
def get_int(input_: str, default=0) -> int:
    try:
        my_num = int(input_)
        return my_num
    except ValueError:
        logging.debug(f'Invalid int {input_} set to default: {default}')
        return default

def get_float(input_: str, default=0.0) -> float:
    try:
        my_num = float(input_)
        return my_num
    except ValueError:
        logging.debug(f'Invalid float {input_} set to default: {default}')
        return default

def post_processing_text(output_text, stop_tokens, denylist = []):
    logging.debug(f"<post_processing_text> output_text: {output_text}")

    filtered_stop_tokens = []
    for token in stop_tokens:
        if token != '':
            filtered_stop_tokens.append(token)

    logging.debug(f"<post_processing_text> stop_tokens: {filtered_stop_tokens}.")

    end_pos = len(output_text)
    logging.debug(f"<post_processing_text>1 end_pos: {end_pos}.")
    for stop_token in filtered_stop_tokens:
        if output_text.find(stop_token) != -1:
            end_pos = min(output_text.find(stop_token), end_pos)

    logging.debug(f"<post_processing_text>2 end_pos: {end_pos}.")
    logging.debug(f"<post_processing_text> text: {output_text}, end_pos: {end_pos}")
    post_processed_text = output_text[:end_pos]
    logging.debug(f"<post_processing_text> input: {output_text}")
    logging.debug(f"<post_processing_text> output: {post_processed_text}")
    start = timeit.default_timer()
    for word in denylist:
        if post_processed_text.find(word) != -1:
            print(f"<post_processing_text> post_processed_text: {post_processed_text}")
            print(f"<post_processing_text> denylist word {word} found, set to empty.")
            post_processed_text = "Sorry, I'm not sure how to answer that question."
            break
    stop = timeit.default_timer()
    print(f"<post_processing_text> time: {stop - start}")
    return post_processed_text


def convert_hf_score_to_logprobs(scores, k, tokenizer):
    results = []
    batch_size = scores[0].shape[0]
    print(f"<convert_hf_score_to_logprobs>: batch size: {batch_size}")

    for i in range(batch_size):
        logprobs = []
        for current_step_score in scores[i:i+1]:
            value, indices = torch.topk(torch.log_softmax(torch.squeeze(current_step_score.float()), dim=-1), k)
            current_logprob = list(zip(tokenizer.convert_ids_to_tokens(indices.tolist()), value.tolist()))
            logprobs.append(current_logprob)
        results.append(logprobs)
    return results

