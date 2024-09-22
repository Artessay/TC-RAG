import os
import json

methods = ['Base', 'CoT', 'BasicRAG', 'Sure', 'SentenceRAG', 'TokenRAG', 'EntityRAG', 'AttnWeightRAG']
datasets = ['CMB', 'MMCU']
models = ['Qwen', 'Xiaobei']

for dataset in datasets:
    for method in methods:
        for model in models:
            row_path = f'result/{method}/{dataset}/{model}/row_test_full.json'
            result_path = f'result/{method}/{dataset}/{model}/raw_test_full.json'
            if not os.path.exists(row_path):
                if not os.path.exists(result_path):
                    print(f"[warning] file {row_path} does not exists")
                continue

            with open(row_path, 'r') as f:
                data = json.load(f)
            sample_data = data[:500]

            with open(result_path, 'w') as f:
                json.dump(sample_data, f, indent=4, ensure_ascii=False)
            
            ex_result_path = f'result/{method}/{dataset}/{model}/ex_test_full.json'
            process_result_path = f'result/{method}/{dataset}/{model}/process_test_full.json'
            if not os.path.exists(ex_result_path):
                if not os.path.exists(process_result_path):
                    print(f"[warning] file {ex_result_path} does not exists")
                continue
            
            with open(ex_result_path, 'r') as f:
                result = json.load(f)
            sample_result = result[:500]

            with open(process_result_path, 'w') as f:
                json.dump(sample_result, f, indent=4, ensure_ascii=False)
            