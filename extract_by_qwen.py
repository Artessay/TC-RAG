import json
import time
import datetime

from microservice import CustomLanguageModel


def extract_answer(args):
    llm = CustomLanguageModel("Aliyun", use_local=False)
    with open(args.result_path, 'r') as f:
        data_list = json.load(f)

    res_list = []
    checkpoint = args.checkpoint
    checkpoint_dir = args.checkpoint_dir
    if checkpoint != 0:
        f = open('{}/ex_test_full_before_{}.json'.format(checkpoint_dir, checkpoint), 'r')
        content = f.read()
        res_list = json.loads(content)
        f.close()

    print(len(data_list))
    for item in data_list:
        question =item["option_str"]
        id = item["id"]
        if id < checkpoint:
            continue

        print("{}/{}".format(id, len(data_list)))
        data = "下面有一些选项和学生的回答，我需要你理解学生的回答，告诉我学生的答案具体哪个或哪些选项\n"+"【选项】\n"+question
        data = data + "\n【学生回答】\n"+item["result"]+"\n请直接回答学生回答的是问题里的哪个或哪些选项(只回答选项的字母，不要回答选项的内容)，不要回答额外的内容。请回答选择的是选项:"
        starttime = datetime.datetime.now()

        if id % 10 == 0:
            json.dump(res_list, open('{}/ex_test_full_before_{}.json'.format(checkpoint_dir, id), 'w'),
                    indent=4, ensure_ascii=False)
            
        for i in range(10):
            try:
                res = llm(data)
                break
            except Exception as e:
                print(e)
                if "Input data may contain inappropriate content" in str(e):
                    res = 'X'
                    break
                if i == 4:
                    exit(0)
                time.sleep(10)

        item["extract_result"] = res
        res_list.append(item)
        endtime = datetime.datetime.now()

        print("Gold answer:", item["answer"])
        print("Pred answer:", res)
        print("use time:", (endtime - starttime).seconds, "s")

    with open(args.ex_result_path, 'w') as f:
        json.dump(res_list, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    from utils import get_args
    from dotenv import load_dotenv
    load_dotenv()

    args = get_args()
    extract_answer(args)