import torch
from torch import optim
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Batch, DataLoader
from sklearn.model_selection import train_test_split
import os
from model import DualTowerModel, W_BCELoss
from utils import predict, top_k_hit_rate
from dataloader import NewGraphQuestionDataset

# Constants
TEST_SIZE = 0.1
RANDOM_STATE = 42
BATCH_SIZE = 100
NUM_EPOCHS = 1000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.01
W_POSITIVE = 10
W_NEGATIVE = 0.5
THRESHOLD = 0.5
HIDDEN_SIZE = 500
EMBEDDING_SIZE = 384

best_validation_f1 = 0.0




def save_checkpoint(state, filename="best_model.pth"):
    torch.save(state, filename)

def predict(model, graph, question_embedding,node_idx_lists):
    """
    Predict the score for each node in the graph given the question embedding.

    Args:
        model: The PyTorch model that takes a graph, a node index, and a question embedding and returns a score for the node.
        graph: A graph structure compatible with your model.
        question_embedding: A tensor representing the embedding of the question.

    Returns:
        scores (torch.Tensor): A tensor of scores for each node in the graph.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    question_embedding = question_embedding.to(device)
    graph.to(device)
    scores = []
    model.eval()
    with torch.no_grad():
      # Assume the graph is represented in a way that each node can be indexed
      for idx in range(len(node_idx_lists)):  # Adjust the range according to how you can index nodes in your graph
          # Your model needs to accept a single node index and a question embedding and return a score
          node_idx = torch.tensor([node_idx_lists[idx]])
          node_idx.to(device)
          score = model(graph, question_embedding.unsqueeze(0), node_idx)
          scores.append(score)

    # Convert list of scores to a tensor
    scores_tensor = torch.tensor(scores, device=device)
    return scores_tensor

def top_10_hit_rate(model, dataset):
    hit_rates = []
    for graph_idx, graph in enumerate(dataset.graphs):
        question_embedding = dataset.question_embeddings[graph_idx]
        labels = dataset.labels_list[graph_idx]
        node_idx_lists = dataset.node_idx_lists_list[graph_idx]

        # Generate scores for each node in the graph
        node_scores = predict(model,graph, question_embedding, node_idx_lists)  # model.predict should be defined to return scores for each node

        # Determine the indices of the top 10 nodes

        top_10_indices = torch.topk(node_scores, 5).indices.tolist() if len(node_idx_lists) >= 10 else torch.topk(node_scores, len(node_idx_lists)).indices.tolist()

        # Find total relevant nodes (labels with 1)
        total_relevant = sum(labels)

        # Count hits
        hits = sum(1 for i in top_10_indices if labels[i] == 1)

        # Calculate hit rate for this graph
        hit_rate = hits / total_relevant if total_relevant > 0 else 0
        hit_rates.append(hit_rate)

        #print(f"Graph {graph_idx + 1}: Hit Rate = {hit_rate:.2f}")

        #print(f"Graph {graph_idx + 1}: Hit Rate = {hit_rate:.2f}")

    return sum(hit_rates)/len(hit_rates)

# Calculate class weights and setup sampler
def calculate_weights_and_sampler(flat_data, indices):
    label_counts = {}
    for idx in indices:
        label = flat_data[idx][2]  # Assuming label is the third item in each tuple
        label_counts[label] = label_counts.get(label, 0) + 1

    total_count = sum(label_counts.values())
    class_weights = {cls: total_count / count for cls, count in label_counts.items()}

    # Generate sample weights only for the training indices
    sample_weights = [class_weights[flat_data[idx][2]] for idx in indices]
    sample_weights = torch.DoubleTensor(sample_weights)

    # Create a sampler for these weights
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return sampler

# Data preparation and loader functions
def flatten_data(new_d):
    flat_data = []
    for graph_idx, graph in enumerate(new_d.graphs):
        for sample_idx, label in enumerate(new_d.labels_list[graph_idx]):
            question_embedding = new_d.question_embeddings[graph_idx]
            node_idx = new_d.node_idx_lists_list[graph_idx][sample_idx]
            flat_data.append((graph, question_embedding, label, node_idx))
    return flat_data

def rebuild_dataset(indices, flat_data):
    graphs = []
    question_embeddings = []
    labels_list = []
    node_idx_lists_list = []
    for idx in indices:
        graph, question_embedding, label, node_idx = flat_data[idx]
        try:
            graph_idx = graphs.index(graph)
        except ValueError:
            graphs.append(graph)
            question_embeddings.append(question_embedding)
            labels_list.append([label])
            node_idx_lists_list.append([node_idx])
        else:
            labels_list[graph_idx].append(label)
            node_idx_lists_list[graph_idx].append(node_idx)
    return NewGraphQuestionDataset(graphs, question_embeddings, labels_list, node_idx_lists_list)

def setup_loaders(train_dataset, val_dataset, train_sampler):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader

# Training and validation routines
def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    total_samples = 0
    for data in train_loader:
        graph, question_embedding, labels, node_idx = data
        graph = graph.to(DEVICE)
        question_embedding = question_embedding.to(DEVICE)
        labels = labels.to(DEVICE).float()
        optimizer.zero_grad()
        outputs = model(graph, question_embedding, node_idx)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.numel()
    return total_loss / total_samples

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_predicted_positives = 0
    total_true_positives = 0
    total_actual_positives = 0
    total_samples = 0
    with torch.no_grad():
        for data in val_loader:
            graph, question_embedding, labels, node_idx = data
            graph = graph.to(DEVICE)
            question_embedding = question_embedding.to(DEVICE)
            labels = labels.to(DEVICE).float()
            outputs = model(graph, question_embedding, node_idx)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            predicted_classes = (outputs > THRESHOLD).float()
            correct = (predicted_classes == labels.unsqueeze(1)).float().sum()
            total_correct += correct.item()
            total_samples += labels.numel()
            true_positives = (predicted_classes * labels.unsqueeze(1)).sum().item()
            predicted_positives = predicted_classes.sum().item()
            actual_positives = labels.sum().item()
            total_true_positives += true_positives
            total_predicted_positives += predicted_positives
            total_actual_positives += actual_positives

    average_loss = total_loss / total_samples
    average_accuracy = total_correct / total_samples
    average_precision = total_true_positives / total_predicted_positives if total_predicted_positives > 0 else 0
    average_recall = total_true_positives / total_actual_positives if total_actual_positives > 0 else 0
    average_f1_score = 2 * (average_precision * average_recall) / (average_precision + average_recall) if (average_precision + average_recall) > 0 else 0

    return average_loss, average_accuracy, average_precision, average_recall, average_f1_score



# Main execution flow
if __name__ == '__main__':
    torch.manual_seed(42)
    model = DualTowerModel(EMBEDDING_SIZE, HIDDEN_SIZE, 1)
    current_directory = os.path.dirname(__file__)

    # Navigate up one level to the parent directory
    parent_directory = os.path.dirname(current_directory)
    dataset_directory = os.path.join(parent_directory, "output")
    dataset_file_path = os.path.join(dataset_directory, "dataset.pth")
    new_d = torch.load(dataset_file_path)  # Load the preprocessed data

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = W_BCELoss(W_POSITIVE, W_NEGATIVE)
    
    flat_data = flatten_data(new_d)
    labels = [x[2] for x in flat_data]  # x[2] corresponds to the label in each tuple
    train_idx, val_idx = train_test_split(range(len(flat_data)), test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels)
    train_dataset = rebuild_dataset(train_idx, flat_data)
    val_dataset = rebuild_dataset(val_idx, flat_data)
    train_sampler = calculate_weights_and_sampler(flat_data, train_idx)  # Create sampler based on the flattened labels list
    train_loader, val_loader = setup_loaders(train_dataset, val_dataset, train_sampler)
    
    model_directory = os.path.join(parent_directory, "output")
    model_file_path = os.path.join(model_directory , "model.pth")

    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss, val_accuracy, val_precision, val_recall, f1 = validate(model, val_loader, criterion)
        average_hit_rate = top_k_hit_rate(model, val_dataset)
        if average_hit_rate > best_validation_hit_rate:
            best_validation_hit_rate = average_hit_rate
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'hit_rate': best_validation_hit_rate,
            }, filename=model_file_path)
        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}, Validation Accuracy = {val_accuracy:.4f}, Validation precision = {val_precision:.4f}, Validation recall = {val_recall:.4f}, Validation f1 = {f1:.4f}, Validation hit-rate = {average_hit_rate:.4f}')

   
