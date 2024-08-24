from microservice import CustomLanguageModel

class Base():
    def __init__(self, args):
        llm_type = args.model_name
        self.model = CustomLanguageModel(llm_type)

    def inference(self, query):
        query = query
        response = self.model(query)
        return response
