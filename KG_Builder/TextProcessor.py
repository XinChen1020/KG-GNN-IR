
import spacy
from typing import List
from fastcoref import spacy_component
import regex as re
import torch

class TextProcessor:
    def __init__(self, model_path='biu-nlp/lingmess-coref', model_architecture='LingMessCoref'):
        self.nlp = spacy.load("en_core_web_sm")

        self.nlp.add_pipe("fastcoref", config={
            'model_architecture': model_architecture,
            'model_path': model_path,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        })

    def batch_coreference_resolution(self, texts: List[str]) -> List[str]:
        return self.nlp.pipe(texts, component_cfg={"fastcoref": {'resolve_text': True}})

    def find_entities_in_sentence(self, sentence: str, entities: List[str]) -> List[str]:
        entity_pattern = r'\b(?:' + '|'.join(map(re.escape, entities)) + r')\b'
        pattern = re.compile(entity_pattern)
        matches = set(pattern.findall(sentence))
        return list(matches)