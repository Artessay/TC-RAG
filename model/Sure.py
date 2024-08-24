import logging

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import BasicRAG

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

class Sure(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def inference(self, query):
        retrieved_passages = self.retrieve(query)

        # candidate generation
        candidate_choices = self._generate_candidate_choices(query, retrieved_passages)
        logger.info(f"candidate choices: {candidate_choices}")

        if len(candidate_choices) == 1:
            return candidate_choices[0]
        assert len(candidate_choices) >= 2

        # limit the number of candidate choices to avoid exceeds max length
        max_candidate_choices_num = 5
        candidate_choices = candidate_choices[:max_candidate_choices_num]

        # Conditional Summarization for Each Candidate
        summarized_passages = self._summarize_for_candidate(query, retrieved_passages, candidate_choices)

        # Final Answer
        answer = self._get_final_answer(query, candidate_choices, summarized_passages)
        return answer

    def _generate_candidate_choices(self, query, retrieved_passages):
        # get raw candidates
        candidate_prompt = "以下是与最后的考试题目相关的文本段落。阅读这些段落后，请从提供两个正确的答案候选。每个答案应该是这种形式：(a) xx, (b) yy，并且每个候选答案不应超过1句话。"
        candidate_prompt += f"\n\n文本段落：\n{retrieved_passages}\n\n考试题目：\n{query}\n\n候选答案：\n"
        candidate_choices = self.model(candidate_prompt)
        logger.info(f"candidate_choices: {candidate_choices}")

        # post process candidates
        divided_candidates = Sure.divide_candidates(candidate_choices)
        choices_candidates = Sure.handle_except(divided_candidates, candidate_choices)
        return choices_candidates

    def _summarize_for_candidate(self, query, retrieved_passages, candidate_choices):
        # Conditional Summarization for Each Candidate
        summarized_passages = []
        for pred_idx, candidate_choice in enumerate(candidate_choices):
            summarize_prompt = retrieved_passages
            summarize_prompt += "\n\n你的任务是扮演一个专业作家的角色。你将根据上面所提供的各个段落中的信息，撰写一个高质量的段落总结来支持对该问题的答案预测。"
            summarize_prompt += "\n现在，让我们开始。请直接输出总结段落的内容，不要回复额外的信息。"
            summarize_prompt += f"\n\n问题：{query}\n候选答案：{Sure.convert_choices_to_texts(candidate_choices)}\n预测答案：({chr(ord('a')+pred_idx)}) {candidate_choice}\n总结：\n"
            summarized_passage = self.model(summarize_prompt)
            summarized_passages.append(summarized_passage)
        return summarized_passages

    def _get_final_answer(self, query, candidate_choices, summarized_passages):
        final_answer_prompt = f"问题：{query}\n\n"
        for pred_idx, summarized_passage in enumerate(summarized_passages):
            final_answer_prompt += f"总结段落({chr(ord('a')+pred_idx)})：[{summarized_passage}]\n"
        final_answer_prompt += f"\n候选答案：{Sure.convert_choices_to_texts(candidate_choices)}\n"
        final_answer_prompt += "请参考每个候选答案的总结段落，给出问题的最终答案的选项。"
        final_answer = self.model(final_answer_prompt)
        logger.info(f"final answer: {final_answer}")

        return final_answer

    def normalize_answer(s: str):
        """
        Normalize the answer to remove the prefix and suffix.
        """
        chrs = [' ', ',', '.']
        while s[0] in chrs:
            s = s[1:]

        while s[-1] in chrs:
            s = s[:-1]
        
        return s

    def divide_candidates(raw_candidate: str):
        res_item = []
        for i in range(4):
            try:
                target_symbol = chr(i + ord('a'))
                idx = raw_candidate.index(f'({target_symbol})')
                if i < 3: 
                    try:
                        next_symbol = chr(i + 1 + ord('a'))
                        idx_next = raw_candidate.index(f'({next_symbol})')
                        res_item.append(Sure.normalize_answer(raw_candidate[idx + len(target_symbol) + 2:idx_next]))
                    except:
                        res_item.append(Sure.normalize_answer(raw_candidate[idx + len(target_symbol) + 2:]))
                        break
                else:
                    res_item.append(Sure.normalize_answer(raw_candidate[idx + len(target_symbol) + 2:]))
            except:
                # do not find ({target symbol})
                res_item.append(Sure.normalize_answer(raw_candidate))
                break
        return res_item

    def handle_except(res_candidates: list, raw_candidates: str):
        if len(res_candidates) == 0:
            return [Sure.normalize_answer(raw_candidates)]
        elif len(res_candidates) == 1:
            new_res_candidate = []
            for split in res_candidates[0].split(','):
                new_res_candidate.append(Sure.normalize_answer(split))
            res_candidates = new_res_candidate
        
        return res_candidates

    def convert_choices_to_texts(choices):
        res = ''
        
        for i, item in enumerate(choices):
            order_txt = '({})'.format(chr(ord('a') + i))
            ith_txt = order_txt + ' ' + item + ' '
            res += ith_txt
        
        return res[:-1]


if __name__ == '__main__':
    from utils import get_args
    args = get_args()
    model = Sure(args)

    raw_candidate = "(a) A, (b) C"
    divided_candidates = Sure.divide_candidates(raw_candidate)
    assert divided_candidates == ["A", "C"]

    question = """
以下是中国医师考试中规培结业考试的一道麻醉科相关的单项选择题，请分析每个选项，并最后给出答案。
关于腋鞘的描述，错误的是
A. 由颈深筋膜的内脏筋膜形成
B. 包绕腋血管和臂丛
C. 喙突下臂丛阻滞穿刺点在喙突下2cm
D. 腋路臂丛阻滞穿刺点在腋窝最高点压住腋动脉搏动，在指尖前方向腋腔顶刺入
E. 向上通颈根部
"""
    print(model.inference(question))