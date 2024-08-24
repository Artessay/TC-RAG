import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM

class BasicGenerator():
    def __init__(self, model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            torch_dtype="auto", 
            device_map="auto",
            trust_remote_code=True,
        ).eval()

        self.special_tokens = self.tokenizer.all_special_ids
    
    def generate(self, input_text, max_new_tokens=4096):
        if isinstance(input_text, str):
            input_text = [{"role": "user", "content": input_text}]
        assert isinstance(input_text, list)

        text = self.tokenizer.apply_chat_template(
            input_text,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer(text, return_tensors="pt")
        
        input_ids = model_inputs.input_ids
        input_ids = input_ids.to(self.model.device)
        
        generated_ids = self.model.generate(
            input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=max_new_tokens,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def generate_all(self, input_text, max_new_tokens=4096, solver="max", use_logprob = True, use_attention=True, use_entropy = True, use_logits=True):
        if isinstance(input_text, str):
            input_text = [{"role": "user", "content": input_text}]
        assert isinstance(input_text, list)
        
        text = self.tokenizer.apply_chat_template(
            input_text,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = model_inputs.input_ids

        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model.generate(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            max_new_tokens = max_new_tokens, 
            return_dict_in_generate = True, 
            output_scores = True,
        )

        generated_tokens = outputs.sequences[:, input_length:]
        text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        
        useful_tokens_index = torch.tensor([i for i, token_ids in enumerate(generated_tokens[0]) if token_ids not in self.special_tokens])
        seqlist = [self.tokenizer.decode(token) for token in generated_tokens[0][useful_tokens_index]]

        def compute_logprob():
            # -log prob
            if use_logprob:
                transition_scores = self.model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True
                )
                logprobs: torch.Tensor = transition_scores[0]
                
                seqlogprobs = logprobs[useful_tokens_index]
                seqlogprobs = [p.item() for p in seqlogprobs]
            else:
                seqlogprobs = None

            return seqlogprobs

        def compute_attention():
            # attention
            if use_attention:
                with torch.no_grad():
                    atten = self.model(generated_tokens, output_attentions=True).attentions[-1][0]
                if solver == "max": 
                    mean_atten, _ = torch.max(atten, dim=1)
                    mean_atten = torch.mean(mean_atten, dim=0)
                elif solver == "avg":
                    mean_atten = torch.sum(atten, dim=1)
                    mean_atten = torch.mean(mean_atten, dim=0)
                    for i in range(mean_atten.shape[0]):
                        mean_atten[i] /= (mean_atten.shape[0] - i)
                elif solver == "last_token":
                    mean_atten = torch.mean(atten[:, -1], dim=0)
                else:
                    raise NotImplementedError
                    
                seqattns = mean_atten[useful_tokens_index]
                seqattns = [p.item() for p in seqattns]
            else:
                seqattns = None

            return seqattns

        def compute_entropy():
            # entropy
            if use_entropy:
                stacked_scores = torch.stack(outputs.scores, dim=0).squeeze()
                softmax_probs = torch.softmax(stacked_scores, dim=-1)
                entropies = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-10), dim=-1)
                
                seqentropies = entropies[useful_tokens_index]
                seqentropies = [p.item() for p in seqentropies]
            else:
                seqentropies = None 

            return seqentropies

        def compute_logits():
            # logits
            if use_logits:
                stacked_scores = torch.stack(outputs.scores, dim=0).squeeze()
                softmax_probs = torch.softmax(stacked_scores, dim=-1)

                seqlogits = softmax_probs[useful_tokens_index]
                seqlogits = [p.tolist() for p in seqlogits]
            else:
                seqlogits = None

            return seqlogits

        seqlogprobs, seqattns, seqentropies, seqlogits = \
            compute_logprob(), compute_attention(), compute_entropy(), compute_logits()

        return {
            'text': text,
            'tokens': seqlist,
            'logprobs': seqlogprobs,
            'attentions': seqattns,
            'entropies': seqentropies,
            "logits": seqlogits,
        }

class LoraGenerator(BasicGenerator):
    def __init__(self, model_name_or_path, lora_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            lora_path, 
            torch_dtype="auto", 
            device_map="auto",
            trust_remote_code=True,
        ).eval()

        self.special_tokens = self.tokenizer.all_special_ids
        

if __name__ == "__main__":
    # model = BasicGenerator("Qwen/Qwen2-7B-Instruct")
    # model = BasicGenerator("/home/qrh/data/model/Qwen/Qwen1.5-32B-Chat")
    model = LoraGenerator(
        "/home/qrh/data/model/Qwen/Qwen1.5-32B-Chat",
        "/home/qrh/data/model/Qwen/Qwen1.5-32B-Chat-lora-medical"
    )

    # prompt = "What is the capital of France?"
    prompt = "What is the official name of the United Kingdom?"
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    text = model.generate(messages, max_new_tokens=10)
    print('-' * 10)
    print(text)

    result = model.generate_all(messages, max_new_tokens=10)
    
    print('-' * 10)
    print(result['text'])
    print(result['tokens'])
    print(result['logprobs'])
    print(result['attentions'])
    print(result['entropies'])
    print(len(result['logits']), len(result['logits'][0]) if len(result['logits']) > 0 else 0)
