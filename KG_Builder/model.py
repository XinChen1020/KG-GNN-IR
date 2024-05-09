from torch_geometric.nn import GCNConv,SAGEConv
import torch
import torch.nn.functional as F
import torch.nn as nn

class W_BCELoss(nn.Module):
    def __init__(self, w_p=10.0, w_n=0.5):
        super(W_BCELoss, self).__init__()
        self.w_p = w_p
        self.w_n = w_n
        self.loss_fn = nn.BCELoss(reduction='none')
        
    def forward(self, probabilities, labels):
        probabilities = probabilities.squeeze(1)
        labels = labels.float()
        loss = self.loss_fn(probabilities, labels)
        weights = labels * self.w_p + (1 - labels) * self.w_n
        weighted_loss = torch.mean(weights * loss)
        return weighted_loss


class DualTowerModel(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(DualTowerModel, self).__init__()
        # Replace GraphSAGE with GCN
        self.gcn = SAGEConv(embedding_dim, hidden_dim)

        self.norm_g = nn.LayerNorm(hidden_dim)

        self.text_fc = torch.nn.Linear(embedding_dim, hidden_dim)
        self.text_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.norm_t1 = nn.LayerNorm(hidden_dim)
        self.norm_t2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.5)

        self.question_fc = torch.nn.Linear(embedding_dim, hidden_dim)
        self.norm_q = nn.LayerNorm(hidden_dim)

        self.final_fc = torch.nn.Linear(hidden_dim * 3, hidden_dim)
        self.final_fc2 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.final_fc3 = torch.nn.Linear(hidden_dim, output_dim)
        self.norm_final = nn.LayerNorm(hidden_dim)

    def forward(self, data, question_embedding, idx):
        # Process graph features with GCN
        structural_features = F.relu(self.norm_g(self.gcn(data.x, data.edge_index)))
        structural_features = structural_features[idx]  # Apply indexing to select relevant node features
        
        # Process text features
        text_features = F.relu(self.norm_t1(self.text_fc(data.x[idx])))
        text_features = self.dropout(F.relu(self.norm_t2(self.text_fc2(text_features))))

        # Process question features
        question_features = F.relu(self.norm_q(self.question_fc(question_embedding)))

        # Combine features
        combined_features = torch.cat([structural_features, text_features, question_features], dim=1)
        combined_features = F.relu(self.norm_final(self.final_fc(combined_features)))
        combined_features = torch.cat([combined_features, question_features], dim=1)
        combined_features = F.relu(self.final_fc2(combined_features))
        output = torch.sigmoid(self.final_fc3(combined_features))

        return output
