import gensim.downloader as api
from gensim.models import KeyedVectors

class ModelLoader:
    def __init__(self):
        self.models = {}

    def load_model(self, model_name: str) -> KeyedVectors:
        """Load a pre-trained model."""
        if model_name not in self.models:
            print(f"Loading {model_name} model...")
            if model_name == 'word2vec':
                self.models[model_name] = api.load("word2vec-google-news-300")
            elif model_name == 'glove':
                self.models[model_name] = api.load('glove-wiki-gigaword-300')
            elif model_name == 'fasttext':
                self.models[model_name] = api.load("fasttext-wiki-news-subwords-300")
            print(f"{model_name.capitalize()} model loaded.")
        return self.models[model_name]