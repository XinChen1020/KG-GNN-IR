
from typing import List, Set
from transformers import pipeline
from HotpotqQADataStore import HotpotqQADataStore
from TextProcessor import TextProcessor
from EmbeddingManager import EmbeddingManager
from WikiReference import WikiAPI
import numpy as np
import os
import json
from typing import List, Set
from transformers import pipeline
import numpy as np



class GraphBuilder:
    def __init__(self, model_name = 'Babelscape/mrebel-large', tokenizer_name = 'Babelscape/mrebel-large'):
        self.triplet_extractor = pipeline('text2text-generation', model=model_name, tokenizer=tokenizer_name, device_map="auto", batch_size=10)
        self.text_processor = TextProcessor()
        self.embedding_manager = EmbeddingManager()
        self.wiki_api = WikiAPI()

    @classmethod
    def get_entity(self,kgs):
        return list(set(entity for kg in kgs for triplet in kg for entity in (triplet[0], triplet[2])))

    @classmethod
    def get_relations(self,kgs):
        return list(set(triplet[1] for kg in kgs for triplet in kg))
    
    def batch_kg_generation(self, texts):

        # Coreference resolution and triplet extraction logic here...
        texts = self.text_processor.batch_coreference_resolution(texts)
        texts = self.triplet_extractor.tokenizer.batch_decode([encoded_seq["generated_token_ids"] for encoded_seq in self.triplet_extractor([t._.resolved_text for t in texts],
                                                                decoder_start_token_id=250058,
                                                                return_tensors=True,
                                                                return_text=False)])

        return [self.extract_triplets_typed(text) for text in texts]
    
    def extract_triplets_typed(self, text: str) -> List[Set]:

        kg = []
        triplets = []
        relation = ''
        text = text.strip()
        current = 'x'
        subject, relation, object_, object_type, subject_type = '','','','',''

        for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").replace("tp_XX", "").replace("__en__", "").split():
            if token == "<triplet>" or token == "<relation>":
                current = 't'
                if relation != '':
                    #triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
                    triplets.append((subject.strip(), relation.strip(), object_.strip()))
                    relation = ''
                subject = ''
            elif token.startswith("<") and token.endswith(">"):
                if current == 't' or current == 'o':
                    current = 's'
                    if relation != '':
                        #triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
                        triplets.append((subject.strip(), relation.strip(), object_.strip()))
                    object_ = ''
                    subject_type = token[1:-1]
                else:
                    current = 'o'
                    object_type = token[1:-1]
                    relation = ''
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token
        if subject != '' and relation != '' and object_ != '' and object_type != '' and subject_type != '':
            #triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})

            triplets.append((subject.strip(), relation.strip(), object_.strip()))

        return triplets

    def should_call_api(self, cosine_sim, high_threshold=0.8, low_threshold=0.15):
    # No API call if the similarity is above the high threshold or below the low threshold
      if cosine_sim > high_threshold or cosine_sim < low_threshold:
          return False
      return True

    
    def identify_matches(self, entities, embeddings, high_threshold=0.85, low_threshold=0.15):
        # Normalize embeddings and compute cosine similarity matrix
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        cosine_sim = np.dot(embeddings_norm, embeddings_norm.T)

        entity_to_representative = {}
        matches = {}

        for i, entity_i in enumerate(entities):
            for j, entity_j in enumerate(entities):
                if i >= j:  # Avoid redundant comparisons and self-comparisons
                    continue

                # Determine the necessity of an API call
                if self.should_call_api(cosine_sim[i, j], high_threshold, low_threshold):
                    # Make API calls if necessary
                    if entity_i not in matches:
                        matches[entity_i] = self.wiki_api.call_wiki_api(entity_i)
                    if entity_j not in matches:
                        matches[entity_j] = self.wiki_api.call_wiki_api(entity_j)

                    # Compare API results and group entities
                    if (matches[entity_i] == matches[entity_j]) and matches[entity_i]:
                        rep = min(entity_i, entity_j)  # Optionally choose a consistent representative
                        entity_to_representative[entity_i] = rep
                        entity_to_representative[entity_j] = rep
                else:
                    # If no API call is needed, assume grouping based on thresholds
                    if cosine_sim[i, j] > high_threshold:
                        rep = min(entity_i, entity_j)
                        entity_to_representative[entity_i] = rep
                        entity_to_representative[entity_j] = rep
                    elif cosine_sim[i, j] < low_threshold:
                        entity_to_representative[entity_i] = entity_i
                        entity_to_representative[entity_j] = entity_j

        return entity_to_representative

    
    def merge_graphs(self, kgs, matches):
        merged_kg = []
        for kg in kgs:
            for s, p, o in kg:
                s_new = matches.get(s, s)
                o_new = matches.get(o, o)
                merged_kg.append((s_new, p, o_new))
        return merged_kg
    
    def attach_related_entities(self, entities, embeddings, threshold=0.65):

        related_triplets = []

        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Compute cosine similarity matrix
        cosine_sim = np.dot(embeddings_norm, embeddings_norm.T)

        # Find index pairs where cosine similarity is above the threshold
        high_sim_indices = np.where(np.triu(cosine_sim, k=1) > threshold)

        for i, j in zip(*high_sim_indices):
            entity_i = entities[i]
            entity_j = entities[j]


            related_triplets.append((entity_i, 'related', entity_j))

        return related_triplets
    
    def attach_sentences_to_graph(self, entities, sentences, entity_embeddings, sentence_embeddings):
        if entity_embeddings is None or sentence_embeddings is None:
            print("Error: Missing embeddings.")
            return []

        num_entities = len(entities)
        num_sentences = len(sentences)

        embeddings = np.concatenate((sentence_embeddings,entity_embeddings), axis=0)

        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Compute cosine similarity matrix
        cosine_sim = np.dot(embeddings_norm, embeddings_norm.T)


        attachment = []

        for i in range(num_sentences):
            #similar_entities = []
            similar_entities = self.text_processor.find_entities_in_sentence(sentences[i], entities)

            if not similar_entities:

                max_similar_id = np.argmax(cosine_sim[i, num_sentences:])
                attachment.append((sentences[i], 'related to', entities[max_similar_id]))

            else:
                for entity in similar_entities:
                    attachment.append((sentences[i], 'mentions', entity))

        return attachment
    
    def attach_sentences_to_title(self,sentence_nodes):
        attachment = []
        for node in sentence_nodes:
            attachment.append((node.text, 'title',node.metadata['title']))
        return attachment

    def final_graph_generation(self, sentence_nodes, doc_nodes):
        

        kgs = self.batch_kg_generation([node.text for node in doc_nodes])


        sentences = [node.text for node in sentence_nodes]
        entities = self.get_entity(kgs)
        entity_embeddings = self.embedding_manager.get_entity_embeddings(entities)
        sentence_embeddings = self.embedding_manager.get_sentence_embeddings(sentences)

        related_entities_attachment = self.attach_related_entities(entities, entity_embeddings)
        title_attachment = self.attach_sentences_to_title(sentence_nodes)
        sentence_attatchment = self.attach_sentences_to_graph(entities, sentences, entity_embeddings, sentence_embeddings)

        matches = self.identify_matches(entities, entity_embeddings)

        merged_kg = self.merge_graphs(kgs, matches)

        for s, p, o in related_entities_attachment:
            s_new = matches.get(s, s)
            o_new = matches.get(o, o)
            if s_new != o_new:
              merged_kg.append((s_new, p, o_new))

        for sentence, p, o in sentence_attatchment:

            o_new = matches.get(o, o)
            merged_kg.append((sentence, p, o_new))

        for sentence, p, o in title_attachment:
            merged_kg.append((sentence, p, o))

        return merged_kg

if __name__ == '__main__':
    # Initialize the GraphBuilder class
    graph_builder = GraphBuilder()
    
    # Get the current script's directory
    current_directory = os.path.dirname(__file__)

    # Navigate up one level to the parent directory
    parent_directory = os.path.dirname(current_directory)

    # Example usage on  Hotpot qa
    
    dataset_directory = os.path.join(parent_directory, "dataset")
    dataset_file_path = os.path.join(dataset_directory, "hotpot_train_v1.1.json")
    with open(dataset_file_path, encoding = "utf-8") as test:
        outputs = json.load(test)
    
    datastore = HotpotqQADataStore(outputs)

    graph_triplets = graph_builder.final_graph_generation(datastore.sentence_nodes[0], datastore.doc_nodes[0])
    

    filename = "kgs.json"

    # Path to the target directory at the same level as the parent
    target_directory = os.path.join(parent_directory, "output")

    # Create the target directory if it doesn't exist
    os.makedirs(target_directory, exist_ok=True)

    # Full path to the file
    file_path = os.path.join(target_directory, filename)

    with open(file_path, 'a') as f:
        f.write(json.dumps(graph_triplets) + '\n')