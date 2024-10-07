# TC-RAG: Turing-Complete RAG

Welcome to the official GitHub repository for TC-RAG (Turing-Complete RAG)!

This is the official code for paper: **TC-RAG:Turing-Complete RAG's Case study on Medical LLM Systems** (https://arxiv.org/abs/2408.09199)


## Overview

In the pursuit of enhancing medical Large Language Models (LLMs), Retrieval-Augmented Generation (RAG) emerges as a promising solution to mitigate issues such as hallucinations, outdated knowledge, and limited expertise in highly specialized queries. However, existing approaches to RAG fall short by neglecting system state variables, which are crucial for ensuring adaptive control, retrieval halting, and system convergence. This paper introduces the Turing-Complete RAG, a novel framework that addresses these challenges by incorporating a Turing Complete System to manage state variables, thereby enabling more efficient and accurate knowledge retrieval. By leveraging a memory stack system with adaptive retrieval, reasoning, and planning capabilities, Turing-Complete RAG not only ensures the controlled halting of retrieval processes but also mitigates the accumulation of erroneous knowledge via Push and Pop actions. Our extensive experiments on real-world medical datasets demonstrate the superiority of Turing-Complete RAG over existing methods in accuracy by over 7.20\%.



## Install Environment

We use conda to manage the environment.
Please refer to the following steps to install the environment:

```sh
conda create -n TCRAG python=3.11 -y
conda activate TCRAG
pip install -r requirements.txt
python -m spacy download zh_core_web_trf
```

## Setup Basic Config for Large Language Model

TC-RAG mainly supports Large Language Models Qwen, which is a series of transformer-based large language models by Alibaba Cloud.

### Deploy a Large Language Model in Local

If you want to deploy a large language model in local, just change the `model_path` in `microservice/config.py` to actual path of your model. The variable `model_path` should be the path to the directory containing the model files. 
Besides, if you want to use a finetuned large language model with lora weights, you can set `lora_model_path` to the path of the directory containing the lora weight files.

### Use a Large Language Model in Cloud

Some baseline methods do not require treating the LLM as a whitebox system. Therefore, we provide a simple interface to use a LLM in cloud. Just chagne the URL defined in `microservice/CustomLanguageModel.py` to the URL of your own deployed LLM, or use dashscope to call the LLM in cloud. If you use dashscope, please set the `OPENAI_API_KEY` environment variable to your own API key in `.env` file.

## Running

To run the code, simply execute the following command:

```sh
python main.py
```

And we have provided some arguments to run the code with different baseline methods, different datasets, and different large language models. You can enable all these augments with the following command:

```sh
python main.py --module_name "your_module_name" --model_name "your_model_name" --dataset_name "your_dataset_name"
```

The following table lists all the available arguments, their default values and options for each argument:

| Argument            | Default Value | Options                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
| ------------------ | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| `--module_name`     | `Base`       | `Base`, `CoT`, `Sure`, `BasicRAG`, `TokenRAG`, `EntityRAG`, `SentenceRAG`, `TCRAG`
| `--model_name`      | `Qwen` | `Qwen`(used for local LLM), `Aliyun`(used for cloud LLM), `Xiaobei`(used for finetuned LLM)
| `--dataset_name`    | `CMB` | `CMB`, `MMCU`, `Clin`

## Citation
```sh
@misc{jiang2024tcragturingcompleteragscasestudy,
      title={TC-RAG:Turing-Complete RAG's Case study on Medical LLM Systems}, 
      author={Xinke Jiang and Yue Fang and Rihong Qiu and Haoyu Zhang and Yongxin Xu and Hao Chen and Wentao Zhang and Ruizhe Zhang and Yuchen Fang and Xu Chu and Junfeng Zhao and Yasha Wang},
      year={2024},
      eprint={2408.09199},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2408.09199}, 
}
```
