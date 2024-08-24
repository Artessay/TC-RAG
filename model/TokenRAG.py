import math
import logging
import numpy as np

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SentenceRAG
from microservice import BingSearch, DocumentSearch

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

class TokenRAG(SentenceRAG):
    def __init__(self, args):
        super().__init__(args)

        self.sentence_solver = getattr(args, 'sentence_solver', 'avg')
        self.hallucination_threshold = getattr(args, 'hallucination_threshold', 0.1)
    
    def modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in self.nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        sentence_token_start_index = 0
        for sid, sent in enumerate(sentences):
            pos = 0 # current find position
            sentence_token_end_index = sentence_token_start_index
            while sentence_token_end_index < len(tokens):
                apr = sent[pos:].find(tokens[sentence_token_end_index])
                if apr == -1: # this token is not in the sentence, we got to the end
                    break
                pos = apr + len(tokens[sentence_token_end_index])
                sentence_token_end_index += 1
            
            probs = [
                1 - math.exp(v) 
                for v in logprobs[sentence_token_start_index:sentence_token_end_index+1]
            ]
            probs = np.array(probs)
            
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)

            if p > self.hallucination_threshold: # hallucination
                # keep sentences before hallucination 
                prev = "" if sid == 0 else " ".join(sentences[:sid-1])
                # replace all hallucinated tokens in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                
                for prob, tok in zip(probs, tokens[sentence_token_start_index:sentence_token_end_index+1]):
                    apr = curr[pos:].find(tok)
                    if apr == -1:
                        break
                    apr += pos
                    if prob > self.hallucination_threshold:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(tok):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(tok)
                return prev, curr, True
            sentence_token_start_index = sentence_token_end_index + 1
        
        # No hallucination
        return text, None, False
    
    def inference(self, query):
        text = ""

        max_new_tokens = 512
        max_generate_tokens = 1024
        max_interaction_num = 5
        current_interaction_num = 0

        while True:
            last_len = len(text)

            # generate answer for first time
            prompt = query + " " + text if len(text) > 0 else query
            result = self.model.generate(
                prompt, 
                max_new_tokens=max_new_tokens, 
                use_logprob=True
            )
            new_text, tokens, logprobs = result['text'], result['tokens'], result['logprobs']

            ptext, curr, hallucination = self.modifier(new_text, tokens, logprobs)

            # limit max interaction times
            current_interaction_num += 1
            if current_interaction_num >= max_interaction_num:
                hallucination = False

            if not hallucination:
                text = text.strip() + " " + new_text.strip()
            else:
                if isinstance(self.retriever, DocumentSearch):
                    retrieve_question = curr.replace("[xxx]", " ")
                elif isinstance(self.retriever, BingSearch):
                    query_list = [query, text, ptext]
                    retrieve_question = " ".join(s for s in query_list if len(s) > 0)
                else:
                    raise NotImplemented

                logger.info(f'retrieve questioin: {retrieve_question}')
                docs = self.retrieve(retrieve_question)
                prompt = query
                prompt += f"\n\n以下是一些检索到的和问题相关的信息:\n{docs}"
                
                last_answer = text + " " + ptext.strip()
                if last_answer.strip() != "":
                    prompt += "\n\n请根据检索到的信息和考试题目，继续上次未完成回答的内容。" + \
                        "请不要重复上次已经输出的文本，直接继续回答即可。如果你觉得已经知道最终答案了，请按下面格式输出：\n" + \
                        "答案为：{选项}"
                    prompt += "\n\n上次回答的内容为：\n" + last_answer
                else:
                    prompt += "\n\n请根据检索到的信息和考试题目，一步步思考分析，并给出最终的答案。"
                           
                new_text = self.model(prompt)
                
                text = text.strip() + " " + ptext.strip() + " " + new_text.strip()
            
            # 判断 token 的个数要少于 generate_max_length 
            tokens_count = len(text.split())
            if tokens_count > max_generate_tokens or len(text) <= last_len or \
                "答案为" in text or "答案是" in text or "答案选" in text or not hallucination:
                break
        return text

if __name__ == '__main__':
    from utils import get_args
    args = get_args()
    args.model_name = "Xiaobei"
    rag = TokenRAG(args)
    question = """
以下是中国医师考试中规培结业考试的一道麻醉科相关的单项选择题，请分析每个选项，并最后给出答案。
关于腋鞘的描述，错误的是
A. 由颈深筋膜的内脏筋膜形成
B. 包绕腋血管和臂丛
C. 喙突下臂丛阻滞穿刺点在喙突下2cm
D. 腋路臂丛阻滞穿刺点在腋窝最高点压住腋动脉搏动，在指尖前方向腋腔顶刺入
E. 向上通颈根部
"""
    print(rag.inference(question))
