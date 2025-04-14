# SmalLM

<p align="center">
  <a href="#about">About</a> •
  <a href="#overview">Overview</a> •
  <a href="#tutorials">Tutorials</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>


<div align="center">
  <a href="https://huggingface.co/collections/Azrail/smallm-67fad955d5b50b57ab917f6b">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E" style="vertical-align:middle">
  </a>
  <a href="/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" style="vertical-align:middle">
  </a>
</div>


## About

This project aims to train series of small transformer models for language modeling from scratch.

The main purpose of this repository is to explore and experiment with new approaches to transformers.

## 🚀 Overview  <span id="overview"></span>
This repository offers modular pipelines for Pretraining, Fine-Tuning models, and Alignment.

### ✅ Implemented Features
- **Data Engine**: Scalable dataset preprocessing toolkit
- **Pretraining**: Base model training infrastructure
- **Alignment**:
  - Supervised Fine-Tuning (SFT) workflows
- **Inference**: Model serving templates
- **Model size**: 70M

### 🔧 Active Development
- **Alignment techniques**: DPO, SimPO, Self-Play Fine-Tuning (SPIN), GRPO
- **Reasoning**
- **Extended Modalities**: Cross-modal attention prototypes
- **Model size**: 150M, 350M, 0.5B
- **Adaptation to Russian language domain**

### 🧠 Key Architecture Features
- **GQA**: Grouped Query Attention with KV-cache [`paper`](https://arxiv.org/abs/2305.13245v3)
- **Mixture-of-Experts with Auxiliary Loss-Free balancing**: [`paper`](https://arxiv.org/abs/2408.15664)
- **ALiBi (Attention with Linear Biases)**: [`paper`](https://arxiv.org/abs/2108.12409v2)
- **Rotary Position Embedding (RoPE)**: [`paper`](https://arxiv.org/abs/2104.09864v5)
- **NTK-by-parts RoPE interpolation**: [`paper`](https://arxiv.org/abs/2309.00071), [blog explanation](https://blog.eleuther.ai/yarn/)



## Tutorials (FUTURE) <span id="tutorials"></span>
A series of notebooks is planned with an explanation of the key points from new articles on LLM

<!-- ## Examples (Future) -->


## Installation

Installation may depend on your task.

**For personal use, see the hufggingface [`page`](https://huggingface.co/collections/Azrail/smallm-67fad955d5b50b57ab917f6b).**

**For training:**

0. Install uv - package and virtual enviroments manager [`uv`](https://docs.astral.sh/uv/guides/install-python/).

   a. `standart` version:

   ```bash
   # installing and sync dependencies
   uv sync
   ```

   b. (`optional`) version for maximum speed (only for modern gpu):

   ```bash
   # installing torch with the version of cuda that matches the current installed driver
   uv pip install torch --index-url https://download.pytorch.org/whl/cuXXX

   # installing flash attention for increasing train speed
   uv pip install flash-attn --no-build-isolation
   ```
   if you want to increase the learning rate a little more (using adamw_apex_fused optimizer). Recomended download nvidia apex.
    ```bash
    git clone https://github.com/NVIDIA/apex
    ```
    However, since uv does not support some compilation flags, it is necessary to add them to the apex/setup.py file manualy:
    ```py
    sys.argv.append("--cpp_ext")
    sys.argv.append("--cuda_ext")
    ```
    After all steps run command:
    ```bash
    uv pip install -v --no-cache-dir --no-build-isolation ./apex
    ```

## How To Use

To train a certain stage of model, run the following command:

```bash
uv run src/stages/<stage>.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs/<stage>` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

Stages includes all the necessary steps: build tokenizer, build `<stage>` dataset e.t.c

To use model from source, run the following command:

```py
from src.models import SmalLmForCausalLM, SmalLmConfig
from transformers import AutoTokenizer

config = SmalLmConfig(...)
tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer,
        use_fast=True,
    )

#from zero step
model = SmalLmForCausalLM(config=config)

#or from pretrained state
model = SmalLmForCausalLM.from_pretrained(model_name_or_path=...)

inputs = tokenizer(['some text'], return_tensors='pt')
out = model(**inputs)

# for generation
from transformers import GenerationConfig
generation_config = GenerationConfig(
    max_new_tokens=50,
    use_cache=True,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.2,
)
out = model.generate(**inputs, generation_config=generation_config)
res_text = tokenizer.decode(out)
```
> **⚠️ WARNING**
> Use attn_implementation=flash_attention_2 only for training

Recomennded use for generation hf api [`page`](https://huggingface.co/collections/Azrail/smallm-67fad955d5b50b57ab917f6b):
```py
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(model_name_or_path=...)
tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer,
        use_fast=True,
    )
generation_config = GenerationConfig(
    max_new_tokens=50,
    use_cache=True,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.2,
)

inputs = tokenizer(['some text'], return_tensors='pt')

out = model.generate(**inp, generation_config=generation_config)

res_text = tokenizer.decode(out)
```

## Credits

This repository is based on a heavily modified fork of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
