from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class EmbeddingManager:
    def __init__(self, model_name = "BAAI/bge-small-en-v1.5", cache_folder = "./cache"):
        self.embed_model = HuggingFaceEmbedding(model_name=model_name, cache_folder = cache_folder)

    def get_entity_embeddings(self, entities):
        embeddings = self.embed_model._get_text_embeddings(entities)
        return embeddings

    def get_sentence_embeddings(self, sentences):
        embeddings = self.embed_model._get_text_embeddings(sentences)
        return embeddings
    