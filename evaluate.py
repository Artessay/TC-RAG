import os
import json
from analysis import evaluate

methods = ['Base', 'CoT', 'BasicRAG', 'Sure', 'SentenceRAG', 'TokenRAG', 'EntityRAG', 'AttnWeightRAG']
datasets = ['CMB', 'MMCU']
models = ['Qwen', 'Xiaobei']

f = open('result/output.jsonl', 'w')

for method in methods:
    for model in models:
        for dataset in datasets:
            ex_result_path = f'result/{method}/{dataset}/{model}/process_test_full.json'
            if not os.path.exists(ex_result_path):
                print(f"[warning] file {ex_result_path} does not exists")
                continue

            accuracy = evaluate(ex_result_path)
            result = {
                "dataset_name": dataset,
                "method_name": method,
                "model_name": model,
                "accuracy": accuracy
            }
        
            print(json.dumps(result), file=f)

f.close()