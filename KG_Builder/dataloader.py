import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch.utils.data import Dataset
from HotpotQADataStore import HotpotqQADataStore
from EmbeddingManager import EmbeddingManager
import os
import torch
import json

def triplets_to_graphs(triplets, embed_model):

    G = nx.Graph()
    for s, p, o in triplets:
        G.add_node(s, features=torch.tensor(embed_model._get_text_embedding(s)))
        G.add_node(o, features=torch.tensor(embed_model._get_text_embedding(o)))
        G.add_edge(s, o, relation=p)

    data = from_networkx(G)
    data.x = torch.stack([G.nodes[node]['features'] for node in G.nodes()])

    node_to_index = {node: idx for idx, node in enumerate(G.nodes())}

    return G, data, node_to_index

def labels_list_generation(data, graphs, node_to_index):
    labels = []
    node_idx_lists = []
    for idx, graph in enumerate(graphs):
        num_nodes = len(graph.nodes())
        node_idx_list = []
        label_list = []
        for node in data.sentence_nodes[idx]:
          node_idx = node_to_index[idx].get(node.text)

          if node_idx is not None:
            node_idx_list.append(node_idx)

          if node.id_ in data.supporting_facts_id[idx]:
            label_list.append(1)
          else:
            label_list.append(0)

        node_idx_lists.append(node_idx_list)
        labels.append(label_list)

    return labels, node_idx_lists

def populate_Dataset(data, triplets, embed_model):
    graphs = []
    torch_datas = []
    question_embeddings = torch.tensor(embed_model._get_text_embeddings([q.text for q in data.questions]))
    node_to_index_list = []
    
    for triplet in triplets:
        graph, torch_data, node_to_index = triplets_to_graphs(triplet, embed_model)
        torch_datas.append(torch_data)
        graphs.append(graph)
        node_to_index_list.append(node_to_index)


    labels, node_idx_lists = labels_list_generation(data, graphs, node_to_index_list)

    return NewGraphQuestionDataset(torch_datas, question_embeddings, labels, node_idx_lists)


class NewGraphQuestionDataset(Dataset):
    def __init__(self, graphs, question_embeddings, labels_list, node_idx_lists_list):
        """
        Args:
            graphs (list): Each element is a graph structure, where each graph is used by multiple samples.
            question_embeddings (list of torch.Tensor): Each tensor is a question embedding used by multiple samples.
            labels_list (list of list of int): Each list contains multiple labels, all corresponding to the same graph and embedding.
            node_idx_lists_list (list of list of int): Each list contains multiple node indices, each corresponding to the labels in the same index.
        """
        assert len(graphs) == len(question_embeddings) == len(labels_list) == len(node_idx_lists_list), "All lists must have the same length"

        self.graphs = graphs
        self.question_embeddings = question_embeddings
        self.labels_list = labels_list
        self.node_idx_lists_list = node_idx_lists_list

    def __len__(self):
        # Sum of all labels across all list entries
        return sum(len(labels) for labels in self.labels_list)

    def __getitem__(self, idx):
        # Find the right graph and embedding by iterating over labels_list and counting their cumulative lengths
        cumulative_length = 0
        for graph_index, labels in enumerate(self.labels_list):
            next_cumulative_length = cumulative_length + len(labels)
            if idx < next_cumulative_length:
                item_index = idx - cumulative_length
                graph = self.graphs[graph_index]
                question_embedding = self.question_embeddings[graph_index]
                label = self.labels_list[graph_index][item_index]
                node_idx = self.node_idx_lists_list[graph_index][item_index]
                return graph, question_embedding, label, node_idx
            cumulative_length = next_cumulative_length

        raise IndexError("Index out of range")


if __name__ == "__main__":

    embed_model = EmbeddingManager().embed_model
    kgs = []

        # Get the current script's directory
    current_directory = os.path.dirname(__file__)

    # Navigate up one level to the parent directory
    parent_directory = os.path.dirname(current_directory)

    # Example usage on  Hotpot qa
    datastore = HotpotqQADataStore()
    kg_directory = os.path.join(parent_directory, "output")
    kg_file_path = os.path.join(kg_directory, "kgs.json")
    kgs = []
    with open(kg_file_path, encoding = "utf-8") as kg_file:
          for line in kg_file:
            kgs.append(json.loads(line))
    
    
    datastore = HotpotqQADataStore()
    dataset_directory = os.path.join(parent_directory, "dataset")
    dataset_file_path = os.path.join(dataset_directory, "hotpot_train_v1.1.json")
    with open(dataset_file_path, encoding = "utf-8") as test:
        outputs = json.load(test)
    
    HI = HotpotqQADataStore(outputs)
    
    d = populate_Dataset(HI, kgs, embed_model)

    filename = "dataset.pth"
    
    # Path to the target directory at the same level as the parent
    target_directory = os.path.join(parent_directory, "output")

    # Create the target directory if it doesn't exist
    os.makedirs(target_directory, exist_ok=True)

    # Full path to the file
    file_path = os.path.join(target_directory, filename)
    torch.save(d, file_path)   


