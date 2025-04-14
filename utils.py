import os
import copy
import wandb
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from trainer import Trainer
from unittest.mock import patch
from datasets import load_dataset
from trl.trainer.utils import pad
from accelerate import Accelerator
from vllm import LLM, SamplingParams
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from trl.models import unwrap_model_for_generation
from torchvision.transforms.functional import pil_to_tensor
from accelerate.utils import broadcast_object_list, gather, gather_object
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor

class Utility:

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            assert False

    def set_all_seeds(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def string_connect(*args, split):
        out = ''
        for i, arg in enumerate(args):
            out += arg
            if i != len(args)-1:
                out += split
        return out

    def freeze_model(model):
        for param in model.parameters():
            param.requires_grad=False

    def bfloat_model(model):
        for param in model.parameters():
            if 'float32' in str(param.dtype).lower() or 'float16' in str(param.dtype).lower():
                param.data = param.data.to(torch.bfloat16)

    def repeat(prompt_list, num_gens):
        return [item for item in prompt_list for _ in range(num_gens)]

    def compute_log_probs(model, _inputs, logits_to_keep, temp=0.9, chunk_size=64): 
        # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        logits = model(**_inputs).logits[:, :-1, :]

        input_ids = _inputs.input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:, :] / temp

        """Process in chunks to reduce peak memory"""
        batch_size, seq_len, _ = logits.shape
        log_probs = torch.zeros(batch_size, seq_len, device=logits.device)
        
        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)
            chunk_logits = logits[:, i:end_idx, :]
            chunk_ids = input_ids[:, i:end_idx]
            chunk_log_probs = nn.functional.log_softmax(chunk_logits, dim=-1)
            log_probs[:, i:end_idx] = chunk_log_probs.gather(
                dim=-1, index=chunk_ids.unsqueeze(-1)).squeeze(-1)
            del chunk_logits, chunk_log_probs
            torch.cuda.empty_cache()
        return log_probs

    def preprocess(inputs, processor):

        prompt_list = []
        image_list = []
        qa_index_list = []
        for _input in  inputs:

            # rolling dice to select one QA sample of multiple QA pairs
            qa_index = torch.randint(low=0, high=len(_input['conversations']), size=(1,)).item()

            # System Prompt
            messages = [
                {
                    "role": "system",
                    "content": Utility.string_connect("You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.",
                                                      "Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags.",
                                                      split='\n'),
                }
            ]

            # question prompt
            if 'image' in _input.keys():
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                            },
                            {"type": "text", "text": _input['conversations'][qa_index]['question'].replace('<image>','')}, # <image> token remove
                        ],
                    },
                )
                image_list.append(_input['image'])
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": _input['conversations'][qa_index]['question'],
                    },
                )

            # answer Prompt
            messages.append(
                {
                    "role": "assistant",
                    "content": "Let me solve this step by step.\n<think>"
                }
            )

            # Preparation for inference
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, continue_final_message=True
            )

            # prompt batchify
            prompt_list.append(prompt)

            # qa index batchify
            qa_index_list.append(qa_index)

        return prompt_list, image_list, qa_index_list

    def postprocess(inputs, processor, qa_index_list, completion_texts):

        prompt_list = []
        image_list = []
        for _input, qa_index, ct in  zip(inputs, qa_index_list, completion_texts):

            # System Prompt
            messages = [
                {
                    "role": "system",
                    "content": Utility.string_connect("You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.",
                                                      "Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags.",
                                                      split='\n'),
                }
            ]

            # question prompt
            if 'image' in _input.keys():
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                            },
                            {"type": "text", "text": _input['conversations'][qa_index]['question'].replace('<image>','')}, # <image> token remove
                        ],
                    },
                )
                image_list.append(_input['image'])
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": _input['conversations'][qa_index]['question'],
                    },
                )

            # answer Prompt
            messages.append(
                {
                    "role": "assistant",
                    "content": "Let me solve this step by step.\n<think>"+ct
                }
            )

            # Preparation for inference
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, continue_final_message=True
            )

            # prompt batchify
            prompt_list.append(prompt)
            
        return prompt_list, image_list

    def reward_prompt(output_texts, answers, processor):

        prompt_list = []
        for ot, ans in  zip(output_texts, answers):

            # System Prompt
            messages = [
                {
                    "role": "system",
                    "content": Utility.string_connect("You are a helpful assistant.",
                                                      "For the question, the predicted answer and the true answer will be given.",
                                                      "You should evaluate the predicted answer compared with the true answer after looking at the question.",
                                                      "Once it is finished to evalute the predicted answer, you should give a evaluation score by choosing one of 0, 5, or 10.",
                                                      "If you think it is insufficient answer to the question when comparing the predicted answer with the true answer, then please give the score 0.",
                                                      "Conversly, if you think it feels proper but could be improved to the question, then please give the score 5.",
                                                      "Note that, if you think the predicted answer includes the true answer's main point properly, then give the score 10.",
                                                      split='\n'),
                }
            ]

            # question prompt
            question_text = ot.split('user\n')[1].split('assistant\n')[0].strip()
            assistant_text = ot.split('assistant\n')[1]
            try:
                pred_ans = assistant_text.split('<answer>')[1].split('</answer>')[0]
            except:
                pred_ans = 'No Answer.'
            messages.append(
                {
                    "role": "user",
                    "content": Utility.string_connect("The question is the following:",
                                                      question_text,
                                                      "The predicted answer is:",
                                                      pred_ans,
                                                      "The true answer is:",
                                                      ans,
                                                      "Please say the evaluated score only.",
                                                      "Do not say like this is 0, the evaluation score is 5, or it is really greatr 10!.",
                                                      "And do not say the number's english like zero, five, or ten.",
                                                      "Please do say the number directly like 0, 5, or 10.",
                                                      split='\n')
                },
            )

            # Preparation for inference
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, continue_final_message=False
            )

            # prompt batchify
            prompt_list.append(prompt)
            
        return prompt_list
    
    def functional_reward_fn(model, sampling_params, temperature, processor, output_texts, answers, accel): 
        prompts_text = Utility.reward_prompt(output_texts, answers, processor)

        # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
        all_prompts_text = gather_object(prompts_text)
        
        # group into pairs
        all_multimodal_inputs = []

        # this is a better implementation for vLLM sampling.
        for prompt in all_prompts_text:
            all_multimodal_inputs.append({"prompt": prompt})
        
        # Create sampling params with num_generations
        if accel.is_main_process:
            
            # Single generate call with all prompts
            sampling_params.n=1
            sampling_params.temperature=temperature
            sampling_params.max_tokens=5

            outputs = model.generate(
                all_multimodal_inputs,
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            # Flatten outputs: [prompt1_gen1, prompt1_gen2, ..., prompt2_gen1, prompt2_gen2, ...]
            completion_texts = [out.text.strip() for completion in outputs for out in completion.outputs]
        else:
            completion_texts = [None] * len(all_multimodal_inputs)
        
        accel.wait_for_everyone()

        # broadcast and slice
        completion_texts = broadcast_object_list(completion_texts)
        process_slice = slice(
            accel.process_index * len(prompts_text),
            (accel.process_index + 1) * len(prompts_text),
        )
        completion_texts = completion_texts[process_slice]

        rewards = []
        for ct in completion_texts:
            try:
                r = int(ct.strip())
                rewards.append(max(min(10, r), 0)/5)
            except:
                rewards.append(-1)
        return rewards

    def structural_reward_fn(output_texts): 
        rewards = []
        for response in output_texts:
            
            res = response.split('assistant\n')[1]
            reward = 2
            # <thinkg> number
            if res.count('<think>') != 1: reward -= 0.5
            if res.count('</think>') != 1: reward -= 0.5

            # <answer> number
            if res.count('<answer>') != 1: reward -= 0.5
            if res.count('</answer>') != 1: reward -= 0.5
            
            # no words and sentences between think and answer
            if len(res.split('</think>'))==2:
                if len(res.split('</think>')[1].split('<answer>')[0].strip())!=0: reward -= 0.5
            else:
                reward -= 0.5
                
            # no words and sentences after answer
            if len(res.split('</answer>'))==2:
                if len(res.split('</answer>')[1].strip())!=0: reward -= 0.5
            else:
                reward -= 0.5
            rewards.append(reward)
        return rewards

    # trial and error here, for better weightage between the reward components ... 
    def compute_reward(model, sampling_params, temperature, processor, output_texts, answers, accel): 
        # Rewards: 0 ~ 2
        functional_rewards = Utility.functional_reward_fn(model, sampling_params, temperature, processor, output_texts, answers, accel)
        
        # Rewards: -1 ~ 2
        structural_rewards = Utility.structural_reward_fn(output_texts)

        combined_rewards = []
        for f_score, s_score in zip(functional_rewards, structural_rewards): combined_rewards.append([f_score, s_score])
        return combined_rewards

    def load_vLLM(model_name, accel, gen_dict):
        if accel.is_main_process:
            vllm_device = f"cuda:{accel.num_processes}" # next GPU idx

            world_size_patch = patch(
                "torch.distributed.get_world_size", return_value=1
            )
            profiling_patch = patch(
                "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
                return_value=None,
            )
            with world_size_patch, profiling_patch:
                vllm_model = LLM(
                    model=model_name,
                    device=vllm_device,
                    gpu_memory_utilization=0.8,
                    dtype=torch.bfloat16,
                    enable_prefix_caching=True,
                    enforce_eager=True,
                    max_model_len=10000,
                )
            vllm_sampling_params = SamplingParams(
                top_p=gen_dict['top_p'],
                top_k=gen_dict['top_k'],
                temperature=gen_dict['temperature'],
                repetition_penalty=gen_dict['repetition_penalty'],
                max_tokens=gen_dict['max_new_tokens']
            )
        else:
            vllm_model = None
            vllm_sampling_params = None

        accel.wait_for_everyone()
        return vllm_model, vllm_sampling_params

    def vLLM_generation(vllm_model, vllm_sampling_params, max_new_tokens, model, accel, num_gens, prompts_text, images, batch_size):
        
        # First, have main process load weights if needed
        with unwrap_model_for_generation(
            model,
            accel,
            gather_deepspeed3_params=True, # for DeepSpeed-Zero3
        ) as unwrapped_model:
            state_dict = unwrapped_model.state_dict()
        if accel.is_main_process:
            vllm_model_tuple = (
                vllm_model.llm_engine.model_executor.driver_worker.model_runner.model
            )
            vllm_model_tuple.load_weights(state_dict.items())

        # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
        all_prompts_text = gather_object(prompts_text)
        all_images = gather_object(images)
        
        # group into pairs
        all_multimodal_inputs = []

        # this is a better implementation for vLLM sampling.
        for prompt, image in zip(all_prompts_text, all_images):
            all_multimodal_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image.unsqueeze(0).cpu()}})
        
        # Create sampling params with num_generations
        if accel.is_main_process:
            
            # sampling parameter change
            vllm_sampling_params.n=num_gens
            vllm_sampling_params.temperature=1
            vllm_sampling_params.max_tokens=max_new_tokens

            # Single generate call with all prompts
            outputs = vllm_model.generate(
                all_multimodal_inputs,
                sampling_params=vllm_sampling_params,
                use_tqdm=False,
            )

            # Flatten outputs: [prompt1_gen1, prompt1_gen2, ..., prompt2_gen1, prompt2_gen2, ...]
            completion_texts = [out.text.strip() for completion in outputs for out in completion.outputs]
        else:
            completion_texts = [None] * len(all_multimodal_inputs) * num_gens
        
        # broadcast and slice
        completion_texts = broadcast_object_list(completion_texts)
        process_slice = slice(
            accel.process_index * batch_size * num_gens,
            (accel.process_index + 1) * batch_size * num_gens,
        )
        completion_texts = completion_texts[process_slice]
        return completion_texts