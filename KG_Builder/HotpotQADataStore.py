import json
from llama_index.core import Document
from llama_index.core.schema import TextNode

class HotpotqQADataStore:
  def __init__(self, dataset):

    self.doc_nodes = []
    self.sentence_nodes = []
    self.supporting_facts_id = []
    self.questions = []

    for data in dataset:
      
      self._add_doc_nodes(data['context'])
      self._add_sentence_nodes(data['context'])
      self.questions.append(TextNode(text = data['question']))
      self._add_supporting_facts(data['supporting_facts'])
  
  def _add_doc_nodes(self, context):
    self.doc_nodes.append([Document(text = ''.join(c[1]), doc_id = c[0]) for c in context])
  
  def _add_sentence_nodes(self, context):
    sentence_nodes = []
   
    for paragraph in context:
      id = 0
      for sentence in paragraph[1]:

        if len(sentence.strip()) != 0:
          sentence_nodes.append(TextNode(text = sentence.strip(), id_= paragraph[0] + "_" + str(id), metadata = {"title": paragraph[0]}))
          
        else:
          sentence_nodes.append(TextNode(text = "", id_= paragraph[0] + "_" + str(id), metadata = {"title": paragraph[0]}))

        id += 1

    self.sentence_nodes.append(sentence_nodes)

  def _add_supporting_facts(self, supporting_facts):
    self.supporting_facts_id.append([fact[0] + "_" + str(fact[1]) for fact in supporting_facts])