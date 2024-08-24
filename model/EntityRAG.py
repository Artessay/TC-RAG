import math
import numpy as np

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import TokenRAG


class EntityRAG(TokenRAG):
    def __init__(self, args):
        super().__init__(args)

        self.entity_solver = getattr(args, 'entity_solver', 'first')
    
    def modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in self.nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        entity = []
        for sent in sentences:
            doc = self.nlp(sent)
            li = [ent.text for ent in doc.ents]
            entity.append(li)
        
        belonging = [-1] * len(text)
        pos = 0
        for tid, tok in enumerate(tokens):
            apr = text[pos:].find(tok)
            # assert apr != -1
            if apr == -1:
                break
            apr += pos
            for j in range(pos, apr+len(tok)):
                belonging[j] = tid
            pos = apr + len(tok)
        
        entity_intv = []
        for sid, sent in enumerate(sentences):
            tmp = []
            pos = text.find(sent)
            for ent in entity[sid]:
                apr = text[pos:].find(ent) + pos
                el = belonging[apr]
                er = belonging[apr + len(ent) - 1]
                tmp.append((el, er))
                pos = apr + len(ent)
            entity_intv.append(tmp)

        entity_prob = []
        for ent_itv_per_sent in entity_intv:
            tmp = []
            for itv in ent_itv_per_sent:
                probs = np.array(logprobs[itv[0]:itv[1]+1])
                p = {
                    "avg": np.mean,
                    "max": np.max,
                    "min": np.min,
                    "first": lambda x: x[0] if len(x) > 0 else 0
                }.get(self.entity_solver, lambda x: 0)(probs)
                tmp.append(p)
            entity_prob.append(tmp)

        for sid in range(len(sentences)):
            if len(entity_prob[sid]) == 0:
                continue
            probs = [1 - math.exp(v) for v in entity_prob[sid]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold: # hallucination
                # keep sentences before hallucination 
                prev = "" if sid == 0 else " ".join(sentences[:sid-1])
                # replace all hallucinated entities in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                for prob, ent in zip(probs, entity[sid]):
                    apr = curr[pos:].find(ent) + pos
                    if prob > self.hallucination_threshold:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(ent):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(ent)
                return prev, curr, True
        # No hallucination
        return text, None, False

if __name__ == '__main__':
    from utils import get_args
    args = get_args()
    args.model_name = "Xiaobei"
    rag = EntityRAG(args)
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