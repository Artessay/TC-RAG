from model import Base

class CoT(Base):
    def __init__(self, args):
        super().__init__(args)

    def inference(self, query):
        query = query + " 请一步步思考，然后再给出答案。"
        response = self.model(query)
        return response
