import spacy
from diskcache import Cache
from multiprocessing.connection import Client

doc_rag_host = 'localhost'
doc_rag_port = 63863

class DocumentSearch():
    def __init__(self) -> None:
        self.nlp = None
        self.cache = Cache(".document_search_cache")

    def __call__(self, query, recognize_entity=False):
        if recognize_entity:
            if self.nlp is None:
                self.nlp = spacy.load("zh_core_web_trf")
            entities = [ent.text for ent in self.nlp(query).ents]
            results = "\n\n".join([
                f"Passage #{i+1} [" + self._search_entity(entity) + "]" 
                for i, entity in enumerate(entities)
            ])
            return results
        else:
            return self._search_entity(query)
    
    def _search_entity(self, entity):
        if entity in self.cache:
            return self.cache[entity]
        
        params = {'clear': 0, 'query': entity, 'function_call_type': 'DOC'}
        with Client((doc_rag_host, doc_rag_port)) as conn:
            conn.send(params)
            result = conn.recv()
        
        result_str = "\n".join(result)
        self.cache[entity] = result_str
        return result_str

if __name__ == '__main__':
    doc_search = DocumentSearch()
    print(doc_search('如何制作蛋糕？'))
    print(doc_search('阿里巴巴创立于哪一年？'))
    