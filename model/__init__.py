from .Base import Base
from .BasicRAG import BasicRAG
from .SentenceRAG import SentenceRAG
from .TokenRAG import TokenRAG
from .EntityRAG import EntityRAG

import importlib

def load_agent(method: str, args):
    method_module = importlib.import_module('model.' + method)
    agent = getattr(method_module, method)(args)
    return agent
    