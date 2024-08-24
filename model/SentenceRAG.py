import spacy
import logging

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import BasicRAG

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

class SentenceRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

        spacy_model_name = args.spacy_model_name if "spacy_model_name" in args else "zh_core_web_trf"
        self.nlp = spacy.load(spacy_model_name)
        # self.judge_llm = CustomLanguageModel("Qwen2", use_local=False)
    
    def get_top_sentence(self, text):
        sentences = [sent.text.strip() for sent in self.nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[0] if len(sentences) > 0 else ""

    def get_last_sentence(self, text):
        sentences = [sent.text.strip() for sent in self.nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[-1] if len(sentences) > 0 else "" 
    
    def inference(self, query):
        last_answer = ""
        retrieve_question = query
        config = {
            'max_new_tokens': 256
        }
        stop = ['.', '!', '?', '\n', '。', '！', '？']
        num_sentences = 0

        while True:
            docs = self.retriever(retrieve_question)
            prompt = f"{query}\n\n下面是一些与问题相关的知识，可以供你参考:\n{docs}"
            if last_answer != "":
                prompt += f"\n\n请你继续上次未完成的回答，不要重复之前的已经回答过的内容。如果你觉得已经没有要补充的内容了，请不要输出任何内容。\n上次回答的内容：\n{last_answer}"

            new_answer = self.model(prompt, config=config, stop=stop)
            
            sentence = self.get_top_sentence(new_answer)
            logger.info(f"sentence: {sentence}")
            if sentence == "":
                break

            last_answer = last_answer.strip() + " " + sentence
            retrieve_question = sentence
            num_sentences += 1

            # prompt = "请判断下面回答中是否已经包含了问题的答案\n"
            # prompt += f"【问题】{query}\n【回答】{last_answer}\n"
            # prompt += f"请直接回答“是”或者“否”，不要回答额外的内容。"
            # judge = self.judge_llm(query)
            # "是" in judge or \
            if num_sentences > 5 or \
                "答案为" in last_answer or "答案是" in last_answer or "答案选" in last_answer:
                break
        
        return last_answer

if __name__ == "__main__":
    from utils import get_args
    args = get_args()
    args.model_name = "Qwen2"
    rag = SentenceRAG(args)
    print(rag.inference("如何制作蛋糕？"))