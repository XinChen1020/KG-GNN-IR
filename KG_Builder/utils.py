
import torch

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

def top_k_hit_rate(model, dataset, k = 10):
    hit_rates = []
    for graph_idx, graph in enumerate(dataset.graphs):
        question_embedding = dataset.question_embeddings[graph_idx]
        labels = dataset.labels_list[graph_idx]
        node_idx_lists = dataset.node_idx_lists_list[graph_idx]

        # Generate scores for each node in the graph
        node_scores = predict(model,graph, question_embedding, node_idx_lists)  # model.predict should be defined to return scores for each node

        # Determine the indices of the top 10 nodes

        top_10_indices = torch.topk(node_scores, k).indices.tolist() if len(node_idx_lists) >= k else torch.topk(node_scores, len(node_idx_lists)).indices.tolist()

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