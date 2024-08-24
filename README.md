# TC-RAG: Turing-Complete RAG

Welcome to the official GitHub repository for TC-RAG (Turing-Complete RAG)!



## Overview

In the pursuit of enhancing medical Large Language Models (LLMs), Retrieval-Augmented Generation (RAG) emerges as a promising solution to mitigate issues such as hallucinations, outdated knowledge, and limited expertise in highly specialized queries. However, existing approaches to RAG fall short by neglecting system state variables, which are crucial for ensuring adaptive control, retrieval halting, and system convergence. This paper introduces the Turing-Complete RAG, a novel framework that addresses these challenges by incorporating a Turing Complete System to manage state variables, thereby enabling more efficient and accurate knowledge retrieval. By leveraging a memory stack system with adaptive retrieval, reasoning, and planning capabilities, Turing-Complete RAG not only ensures the controlled halting of retrieval processes but also mitigates the accumulation of erroneous knowledge via Push and Pop actions. Our extensive experiments on real-world medical datasets demonstrate the superiority of Turing-Complete RAG over existing methods in accuracy by over 7.20\%.



## Install Environment

```sh
conda create -n StackAgent python=3.11 -y
conda activate StackAgent
pip install -r requirements.txt
python -m spacy download zh_core_web_trf
```

