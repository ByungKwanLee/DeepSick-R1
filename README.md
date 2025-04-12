# DeepSick-R1

### *"Too much feeling sick while reproducing DeepSeek-R1!!"*

## 📰 Breaking News

- Qwen2.5-VL has been supported!
- The reason why vLLM is the problem is reported for each version.

#### 🚑 Why do we need to see this repository although there are many open-source codes for building DeepSeek-R1?

- My short code lines and a few code files make users happy.
- This code doesn't use huggingface GRPOTrainer class which may bring in frustration because of too much complexities when users customize GRPOTrainer to fit individual research and production.
- This code has only three files (main.py, trainer.py, and utils.py) to know for training, while famous repositories [Open-R1](https://github.com/huggingface/open-r1), [R1-V](https://github.com/Deep-Agent/R1-V), [verl](https://github.com/volcengine/verl), and [TinyZero](https://github.com/Jiayi-Pan/TinyZero) have 1000+ code files, many config files, and too much folders.
- [vLLM](https://github.com/vllm-project/vllm) is applied so that users can generate answer candidates realy fastly.
- Although [vLLM](https://github.com/vllm-project/vllm) is applied, total number of code lines is still short.
- For training with multiple GPU, one GPU will be assigned to vLLM model to generate, and the other GPUs are focusing on training.

**Requirements!!: This repository requires two GPUs at least, because vLLM should be assigned to another GPU in order to separate the training GPU and inference GPU.**

---

## 🚀 Short Talks

- When we train Qwen2-VL-2B-Instruct with 100k QA samples on 2 NVIDIA A100 80GB VRAM, it takes 14 hours to train.
- Once I increase the number of GPUs to 8 NVIDIA A100 80GB VRAM, it takes 4.5 hours to train (Data communications between vLLM GPu and other GPUs may be getting slow down).
- When we choose Qwen2.5-VL-3B-Instrcut, it takes 6 hours to train.
- The GPU memory usage was 40~60GB when unfreezing all MLP parameters in LLM decoder part, where I use 2 batch, 4 number of generations, and 4 GRPO iterations. 
- This repository is dealing with vision language models (VLMs) only, but I believe this code is really easy, so users can easily modify the code for LLM version.
- In the current version, Qwen2.5-VL and latest vLLM are not supported because there is first flash attention issue in latest vLLM version and model parameter access issues. I will let this code updated once it is all resolved.

---


## 🚩 Issue Report for each vLLM version
| Version | Debugging Mode (Accelerate + vLLM on Two GPUs) | Flash Attention2 | ZeRO3 Training (Accelerate-DeepSpeed-ZeRO3 + vLLM on Eight GPUs) | Transformer Version | Qwen2.5-VL Error |
|:-------:|:----------------------------------------------:|:----------------:|:----------------------------------------------------------------:|:-------------------:|:----------------:|
|  0.7.2  |                        O                       |         O        |                                 O                                |         ~4.49.0         |       Many       |
|  **0.7.3**  |                        △ (Can solve by manually editing code)                       |         O        |                                 O                                |        Latest       |       Some       |
|  0.8.0  |                        X                       |         X        |                                 X                                |        Latest       |       Less       |
|  0.8.1  |                        X                       |         X        |                                 X                                |        Latest       |       Less       |
|  0.8.2  |                        X                       |         O        |                                 X                                |        Latest       |       Less       |
|  0.8.3  |                        X                       |         O        |                                 X                                |        Latest       |       Less       |


To make it work for debugging mode, we should manually edit the code in vLLM. The file is vLLM/worker/worker.py
```python
    def init_device(self) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            # torch.cuda.set_device(self.device) # It is the problem, please comment out

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.baseline_snapshot = MemorySnapshot()
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.vllm_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)
```

---

### 🍉 Install

```bash
#!/bin/bash
conda create -n deepsick python=3.12 -y
conda activate deepsick

# install vllm
pip install vllm==0.7.3

# install package
pip install trl wandb debugpy datasets deepspeed accelerate

# flash attention
pip install flash-attn --no-build-isolation
```

---

### 🍲 What to see for understanding

```shell
# Total 831 lines
main.py (292 lines)
trainer.py (108 lines)
utils.py (431 lines)
```

---

### 💻 Training with multi-GPU 

[DeepSpeed](https://github.com/deepspeedai/DeepSpeed)-ZeRO3 is used.
```shell
# ds_accel.yaml is the config file for deepspeed zero3
bash train.sh
```

In this file, you can see the n_gpu. this variable automatically computes the process number for accelerator - DeepSpeed.
Because vLLM and accelerate are not compatible, this simple trick is really helpful to address the compatibility issue.

```bash
#!/usr/bin/env bash
CUDA_DEVICES="0,1,2,3,4,5,6,7"
length=${#CUDA_DEVICES}
n_gpu=$(( ( (length + 1) / 2 ) - 1 ))

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
accelerate launch --config_file ds_accel.yaml \
--num_processes=$n_gpu \
main.py \
--wandb True \
```

