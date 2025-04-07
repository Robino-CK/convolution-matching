import torch
from torch_geometric.nn import HeteroConv, SAGEConv
import torch.nn.functional as F

# Define a heterogeneous GNN model
class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels):
        super().__init__()
        self.convs = {}
        
        for edge_type in metadata[1]:
            self.convs[edge_type] = SAGEConv(-1, hidden_channels, add_self_loops=False)
    
        self.hetero_conv = HeteroConv(self.convs, aggr='sum')
        self.lin = torch.nn.ModuleDict({
            node_type: torch.nn.Linear(hidden_channels, hidden_channels)
            for node_type in metadata[0]
        })
    def forward(self, x_dict, edge_index_dict):
      #  print(edge_index_dict.keys())
        x_dict = self.hetero_conv(x_dict, edge_index_dict)
        
        x_dict = {node_type: F.relu(self.lin[node_type](x)) for node_type, x in x_dict.items()}
        return x_dict
    
    
class ImprovedHeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, x_dict,num_classes, num_layers=2, dropout=0.3):
        super().__init__()
        
        # Extract node types and edge types from metadata
        node_types, edge_types = metadata[0], metadata[1]
        
        # Create embedding layers for each node type with proper dimensions
        self.embeddings = torch.nn.ModuleDict()
        for node_type, feat_dim in {
            'author': x_dict['author'].size(1),
            'paper': x_dict['paper'].size(1),
            'term': x_dict['term'].size(1),
            'conference': x_dict['conference'].size(1)
        }.items():
            self.embeddings[node_type] = torch.nn.Linear(feat_dim, hidden_channels)
        
        # Multiple heterogeneous conv layers for message passing
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                # Use the proper dimensions for source and target nodes
                conv_dict[edge_type] = SAGEConv(
                    hidden_channels, 
                    hidden_channels, 
                 #   add_self_loops=False,
                    normalize=True
                )
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        # Layer normalization for each node type
        self.layer_norms = torch.nn.ModuleDict({
            node_type: torch.nn.LayerNorm(hidden_channels)
            for node_type in node_types
        })
        
        # Output projection layers
        self.output_projs = torch.nn.ModuleDict({
            node_type: torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_channels, num_classes if node_type == 'author' else hidden_channels)
            )
            for node_type in node_types
        })
        
        self.dropout = dropout
        
    def forward(self, x_dict, edge_index_dict):
        # Initial embedding of node features
        x_dict = {node_type: self.embeddings[node_type](x) 
                 for node_type, x in x_dict.items()}
        
        # Apply multiple layers of heterogeneous graph convolutions
        for conv in self.convs:
            # Store previous embeddings for residual connections
            x_dict_prev = {k: v.clone() for k, v in x_dict.items()}
            
            # Apply heterogeneous convolution
            x_dict = conv(x_dict, edge_index_dict)
            
            # Apply layer normalization, non-linearity, dropout and residual connection
            x_dict = {
                node_type: self.layer_norms[node_type](
                    F.relu(x) + x_dict_prev[node_type]  # Residual connection
                )
                for node_type, x in x_dict.items()
            }
            
            # Apply dropout to intermediate representations
            x_dict = {
                node_type: F.dropout(x, p=self.dropout, training=self.training)
                for node_type, x in x_dict.items()
            }
        
        # Final projection for each node type
        output_dict = {
            node_type: self.output_projs[node_type](x)
            for node_type, x in x_dict.items()
        }
        
        # Apply log softmax to author nodes (for classification)
        if 'author' in output_dict:
            output_dict['author'] = F.log_softmax(output_dict['author'], dim=1)
            
        return output_dict
    