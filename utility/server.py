from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_compress import Compress

from microservice.config import model_path, lora_model_path
from microservice import BasicGenerator, LoraGenerator

app = Flask(__name__)
CORS(app)
Compress(app)

qwen_agent = None
med_agent = None

@app.route('/qwen', methods=['GET', 'POST'])
def generate_qwen():
    global qwen_agent
    if qwen_agent is None:
        qwen_agent = BasicGenerator(model_path)

    args: dict = request.get_json()
    input_text = args.get('content')
    max_new_tokens = args.get('max_new_tokens', 2048)
    use_logprob = args.get('use_logprob', True)
    use_attention = args.get('use_attention', True)
    use_entropy = args.get('use_entropy', True)
    use_logits = args.get('use_logits', True)


    result = qwen_agent.generate_all(
        input_text, 
        max_new_tokens, 
        solver="max", 
        use_logprob=use_logprob, 
        use_attention=use_attention, 
        use_entropy=use_entropy, 
        use_logits=use_logits
    )
    return jsonify(result)

@app.route('/xiaobei', methods=['GET', 'POST'])
def generate_xiaobei():
    global med_agent
    if med_agent is None:
        med_agent = LoraGenerator(model_path, lora_model_path)

    args: dict = request.get_json()
    input_text = args.get('content')
    max_new_tokens = args.get('max_new_tokens', 2048)
    use_logprob = args.get('use_logprob', True)
    use_attention = args.get('use_attention', True)
    use_entropy = args.get('use_entropy', True)
    use_logits = args.get('use_logits', True)

    result = med_agent.generate_all(
        input_text, 
        max_new_tokens, 
        solver="max", 
        use_logprob=use_logprob, 
        use_attention=use_attention, 
        use_entropy=use_entropy, 
        use_logits=use_logits
    )
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7777)