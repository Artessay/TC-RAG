import torch
import numpy as np

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import TokenRAG


class AttnWeightRAG(TokenRAG):
    def __init__(self, args):
        super().__init__(args)

        self.retrieve_keep_top_k = 30
    
    def modifier(self, text, tokens, attentions, weight):
        sentences = [sent.text.strip() for sent in self.nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        tid = 0
        for sid, sent in enumerate(sentences):
            tl, tr = tid, tid
            if sid == len(sentences) - 1:
                tl, tr = tid, len(tokens)
            else:
                for i in range(tid + 1, len(tokens)):
                    seq = " ".join(tokens[tl:i])
                    if sent in seq:
                        tr = i
                        break
                tid = tr
                
            attns = attentions[tl:tr]
            attns = np.array(attns) / sum(attns)
            value = [attns[i-tl] * weight[i] * (tr-tl) for i in range(tl, tr)] 
            thres = [1 if v > self.hallucination_threshold else 0 for v in value]
            if 1 in thres:
                # hallucinated
                doc = self.nlp(sent)
                real_words = set(token.text for token in doc if token.pos_ in 
                    ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
                def match(tok):
                    for word in real_words:
                        if word in tok:
                            return True
                    return False
                for i in range(len(thres)):
                    if not match(tokens[tl+i]):
                        thres[i] = 0                
                
                prev = "" if sid == 0 else " ".join(sentences[:sid-1])
                
                return True, prev, tokens[tl:tr], thres
        return False, text, None, None

    def keep_real_words(self, prev_text, curr_tokens, curr_hit):
        curr_text = " ".join(curr_tokens)
        all_text = prev_text + " " + curr_text
        input_ids = self.model.agent.tokenizer.encode(all_text, return_tensors="pt")
        input_length = input_ids.shape[1]
        tokens_tmp = self.model.agent.tokenizer.convert_ids_to_tokens(input_ids[0])

        atten_tmp = self.model.agent.model(input_ids, output_attentions=True).attentions[-1][0]

        # merge tokens
        range_ = [[i, i] for i in range(len(tokens_tmp))]
        tokens = []
        for r in range_:
            tokenseq = "".join(tokens_tmp[r[0]: r[1]+1])
            tokens.append(tokenseq)

        # 获取幻觉词对应的 attention
        curr_st = len(tokens) - len(curr_tokens)
        atten_tmp = torch.mean(atten_tmp, dim=0)
        attns = []
        for r in range_:
            # att = torch.zeros(atten_tmp.shape[0], input_length)
            att = torch.zeros(input_length)
            for i in range(r[0], r[1] + 1):
                if i == 0:
                    continue
                v = atten_tmp[i-1][:r[0]] # 上一位的
                v = v / v.sum()
                t = torch.zeros(input_length)
                t[:r[0]] = v
                att += t
            att /= (r[1] - r[0] + 1)
            # merge token for att
            att = torch.tensor([att[rr[0]:rr[1]+1].sum() for rr in range_])
            attns.append(att)
            
        # 计算每个超过阈值的 token 在前文的 attentions
        forward_attns = torch.zeros(len(tokens))
        hit_cnt = 0
        for i in range(len(curr_hit)):
            if curr_hit[i] == 1:
                forward_attns += attns[curr_st + i]
                hit_cnt += 1
        forward_attns /= hit_cnt
        forward_attns = forward_attns.tolist()

        # 分析词性，保留实词对应的 attns
        doc = self.nlp(all_text)
        real_words = set(token.text for token in doc if token.pos_ in 
                      ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
        
        def match(token):
            for word in real_words:
                if word in token:
                    return True
            return False
        
        real_pairs = []
        for i in range(len(tokens)):
            tok, att = tokens[i], forward_attns[i]
            if i >= curr_st and curr_hit[i - curr_st]:
                continue
            if match(tok):
                real_pairs.append((att, tok, i))
        
        top_k = min(self.retrieve_keep_top_k, len(real_pairs))
        
        real_pairs = sorted(real_pairs, key = lambda x:x[0], reverse=True)
        real_pairs = real_pairs[:top_k]
        real_pairs = sorted(real_pairs, key = lambda x:x[2])
        return " ".join([x[1] for x in real_pairs])
        
    def inference(self, query):
        text = ""
        
        max_new_tokens = 512
        max_generate_tokens = 2048
        max_interaction_num = 5
        current_interaction_num = 0

        while True:
            old_len = len(text)

            # generate answer for first time
            prompt = query + " " + text if len(text) > 0 else query
            result = self.model.generate(
                prompt, 
                max_new_tokens=max_new_tokens, 
                use_logprob=False, 
                use_attention=True,
                use_entropy=True,
                use_logits=False,
            )
            
            new_text, tokens, attns, entropies = \
                result['text'], result['tokens'], result['attentions'], result['entropies']
            weight = entropies # if self.method == "dragin" else [-v for v in logprobs]

            hallucination, ptext, curr_tokens, curr_hit =  self.modifier(new_text, tokens, attns, weight)

            # limit max interaction times
            current_interaction_num += 1
            if current_interaction_num >= max_interaction_num:
                hallucination = False
            
            if not hallucination:
                text = text.strip() + " " + new_text.strip()
            else:
                forward_all = [query, text, ptext]
                forward_all = " ".join(s for s in forward_all if len(s) > 0)

                retrieve_question = self.keep_real_words(
                    prev_text = forward_all, 
                    curr_tokens = curr_tokens, 
                    curr_hit = curr_hit,
                ) 

                docs = self.retrieve(retrieve_question)
                prompt = query
                prompt += f"\n\n以下是一些检索到的和问题相关的信息:\n{docs}"
                
                tmp_li = [query, text, ptext.strip()]
                prompt += " ".join(s for s in tmp_li if len(s) > 0)
                
                new_text = self.model(prompt)
                
                new_text = self.get_top_sentence(new_text)
                tmp_li = [text.strip(), ptext.strip(), new_text.strip()]
                text = " ".join(s for s in tmp_li if len(s) > 0)
                
            
            # 判断 token 的个数要少于 generate_max_length 
            tokens_count = len(self.model.agent.tokenizer.encode(text))
            if tokens_count > max_generate_tokens or len(text) <= old_len or \
                "答案为" in text or "答案是" in text or "答案选" in text or not hallucination:
                break
            
        return text

if __name__ == '__main__':
    from utils import get_args
    args = get_args()
    args.model_name = "Xiaobei"
    rag = AttnWeightRAG(args)
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
