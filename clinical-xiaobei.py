import json
from datetime import datetime
from data import load_dataset
from xiaobei.SAMAAgent import Agent
from utils import calculate_bleu, calculate_rouge, seed_everything

def run_clinical(args):
    assert args.dataset_name == 'Clin'

    seed_everything(args.seed)

    agent = Agent()
    data_list = load_dataset(args.dataset_name)

    data = []
    checkpoint_dir = args.checkpoint_dir
    for id, item in enumerate(data_list):
        print(id+1, '/', len(data_list))

        question = item['question']
        gold_answer = item['answers']
        # pred_answer = agent.inference(question)
        start_time = datetime.now()
        pred_answer = agent.whitebox_pop_react_executor(text=question, history=[])[0]
        end_time = datetime.now()
        print("use time:", (end_time - start_time).seconds, "s")

        data.append({
            'question': question,
            'answers': gold_answer,
            'predict': pred_answer
        })

        # if id + 1 % 10 == 0:
        with open('{}/checkpoint_{}.json'.format(checkpoint_dir, id+1), 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


    with open(args.result_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def evaluate_clinical(args):
    assert args.dataset_name == 'Clin'

    with open(args.result_path, 'r') as f:
        data = json.load(f)

    # 创建空列表来存储answers和llmans
    ans_list = []
    llm_ans_list = []

    # 提取answers和llmans到列表中
    for item in data:
        ans_list.append(item['answers'])
        llm_ans_list.append(item['predict'])

    # 提取标签和预测输出所在的列
    labels_column = ans_list
    predictions_column = llm_ans_list

    # 存储计算得到的值
    bleu_1_calculated_values = []
    bleu_4_calculated_values = []
    rouge_recall_calculated_values = []

    # 逐行处理
    for label, prediction in zip(labels_column, predictions_column):
        # 进行你的计算，这里只是简单的示例，可以根据需要进行修改
        bleu_score = calculate_bleu([label], prediction)
        # 计算 ROUGE 分数
        rouge_scores = calculate_rouge(' '.join(list(label)), ' '.join(list(prediction)))

        # 打印结果
        bleu_1_calculated_values.append(bleu_score['BLEU-1'])
        bleu_4_calculated_values.append(bleu_score['BLEU-4'])
        rouge_recall_calculated_values.append(rouge_scores['rouge-l']['r'])

    # 计算平均值
    mean_value_bleu_1 = sum(bleu_1_calculated_values) / len(bleu_1_calculated_values)
    mean_value_bleu_4 = sum(bleu_4_calculated_values) / len(bleu_4_calculated_values)
    mean_value_rouge_recall = sum(rouge_recall_calculated_values) / len(rouge_recall_calculated_values)

    # 打印结果
    print("BLEU-1 score:", mean_value_bleu_1)
    print("BLEU-4 score:", mean_value_bleu_4)
    print("ROUGE scores:", mean_value_rouge_recall)

if __name__ == '__main__':
    from utils import get_args
    args = get_args()
    run_clinical(args)
    evaluate_clinical(args)
