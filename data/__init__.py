import json

def load_dataset(dataset_name):
    if dataset_name == 'CMB':
        dataset_path = 'data/CMB-500.json'
    elif dataset_name == 'MMCU':
        dataset_path = 'data/MMCU_test.json'
    elif dataset_name == 'Clin':
        dataset_path = 'data/CMB-Clin.json'
    else:
        raise ValueError('Invalid dataset name')
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    return dataset


def get_query(data):
    data["exam_type"] = data["exam_type"]
    data["exam_class"] = data["exam_class"]
    data["exam_subject"] = data["exam_subject"]
    data["question_type"] = data["question_type"]
    data["question"] = data["question"]
    data["option_str"] = "\n".join(
        [f"{k}. {v}" for k, v in data["option"].items() if len(v) > 0 and v!=" "]
    )

    query_prompt = "以下是中国{exam_type}中{exam_class}考试的一道{exam_subject}相关的{question_type}，请分析每个选项，并最后给出答案。\n{question}\n{option_str}"
    query = query_prompt.format_map(data)
    
    return query
