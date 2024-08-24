#模型下载
from modelscope import snapshot_download
from microservice.config import model_path
model_dir = snapshot_download('Qwen/Qwen1.5-32B-Chat', local_dir=model_path)