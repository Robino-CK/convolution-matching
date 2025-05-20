import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv
import torch.nn.functional as F
from dgl.nn import RelGraphConv
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
    def __init__(self, metadata, hidden_channels, x_dict,num_classes, target_feat="author", num_layers=2, dropout=0.3, with_non_linear = True):
        
        super().__init__()
        self.target_feat = target_feat
        # Extract node types and edge types from metadata
        node_types, edge_types = metadata[0], metadata[1]
        
        # Create embedding layers for each node type with proper dimensions
        self.embeddings = torch.nn.ModuleDict()
        for node_type, feat_dim in { i: x_dict[i].size(1) for i in  x_dict.keys()}.items():
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
        if with_non_linear:
            # Output projection layers
            self.output_projs = torch.nn.ModuleDict({
                node_type: torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden_channels, num_classes if node_type == target_feat else hidden_channels)
                )
                for node_type in node_types
            })
        else:
            # Output projection layers
            self.output_projs = torch.nn.ModuleDict({
                node_type: torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden_channels, num_classes if node_type == target_feat else hidden_channels)
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
        if self.target_feat in output_dict:
            output_dict[self.target_feat] = F.log_softmax(output_dict[self.target_feat], dim=1)
            
        return output_dict
    
from dgl.nn import HeteroGraphConv, GraphConv

# Define one R-GCN layer using HeteroGraphConv
class RGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, etypes):
        super().__init__()
        self.conv = HeteroGraphConv({
            etype: GraphConv(in_feats, out_feats)
            for etype in etypes
        })

    def forward(self, g, inputs):
        return self.conv(g, inputs)

# R-GCN model with two layers
class RGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, etypes):
        super().__init__()
        self.layer1 = RGCNLayer(in_feats, hidden_feats, etypes)
        self.layer2 = RGCNLayer(hidden_feats, out_feats, etypes)

    def forward(self, g, inputs):
        h = self.layer1(g, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.layer2(g, h)
        return h
    
    
import torch
from torch_geometric.nn import Linear
import torch.nn.functional as F

class HeteroGCNCond(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers,
                  edge_types, node_types, target_node_type, alpha=1):
        super().__init__()
        self.alpha = alpha
        self.num_layers = num_layers
        self.target_node_type = target_node_type
        
        self.lins = torch.nn.ModuleDict()
        for node_type in node_types:
            self.lins[node_type] = torch.nn.ModuleList()
            self.lins[node_type].append(Linear(-1, hidden_channels,bias=False))
            for _ in range(num_layers-1):
                self.lins[node_type].append(Linear(hidden_channels, hidden_channels,bias=False))
        self.out_lin = Linear(hidden_channels, out_channels)
    def reset_parameters(self,init_list=None):
        if init_list is None:
            for node_type in self.lins.keys():
                for lin in self.lins[node_type]:
                    lin.reset_parameters()
            self.out_lin.reset_parameters()
        else:
            i = 0
            for self_p in self.parameters():
                if self_p.dim()==2:
                    self_p.data.copy_(init_list[i])
                    i += 1
            
    def forward(self, x_dict, adj_t_dict, get_embeddings=False):
        h_dict = {}
        for l in range(self.num_layers):
            for node_type in x_dict.keys():
                if l==0:
                    h_dict[node_type] = F.relu_(self.lins[node_type][l](x_dict[node_type]))
                else:
                    h_dict[node_type] = F.relu_(self.lins[node_type][l](h_dict[node_type]))
            out_dict = {node_type: [self.alpha*x] for node_type,x in h_dict.items()}#
            for edge_type, adj_t in adj_t_dict.items():
                src_type, _, dst_type = edge_type
                out_dict[dst_type].append(adj_t @ h_dict[src_type])
            for node_type in x_dict.keys():
                h_dict[node_type] = torch.mean(torch.stack(out_dict[node_type],dim=0), dim=0)#.relu_()
        
        target_logits = self.out_lin(h_dict[self.target_node_type])
        if get_embeddings:
            h_dict = {node_type:h for node_type,h in h_dict.items()}
            return target_logits, h_dict
        else:
            return target_logits


import torch
from torch_geometric.nn import Linear
import torch.nn.functional as F

class HeteroSGC(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers,
                  node_types, edge_types, target_node_type, num_lins=1, alpha=0.01):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.alpha = alpha
        self.num_layers = num_layers
        self.num_lins = num_lins
        self.target_node_type = target_node_type
        
        self.in_lin_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            self.in_lin_dict[node_type] = torch.nn.ModuleList()
            self.in_lin_dict[node_type].append(Linear(-1, hidden_channels))
            for _ in range(num_lins-1):
                self.in_lin_dict[node_type].append(Linear(hidden_channels, hidden_channels))
        self.out_lin = Linear(hidden_channels, out_channels)
    def reset_parameters(self,init_list=None):
        if init_list is None:
            for node_type in self.in_lin_dict.keys():
                for lin in self.in_lin_dict[node_type]:
                    lin.reset_parameters()
            self.out_lin.reset_parameters()
        else:
            i = 0
            for self_p in self.parameters():
                if self_p.dim()==2:
                    self_p.data.copy_(init_list[i])
                    i += 1
            
    def forward(self, x_dict, adj_t_dict, get_embeddings=False):
        h_dict = {}
        for node_type in x_dict.keys():
            h_dict[node_type] = self.in_lin_dict[node_type][0](x_dict[node_type]).relu_()
            for lin in self.in_lin_dict[node_type][1:]:
                h_dict[node_type] = lin(h_dict[node_type]).relu_()
        for l in range(self.num_layers):
            out_dict = {node_type: [self.alpha*x] for node_type,x in h_dict.items()}
            # out_dict = {node_type: [] for node_type,x in h_dict.items()}
            for edge_type, adj_t in adj_t_dict.items():
                src_type, _, dst_type = edge_type
                out_dict[dst_type].append(adj_t @ h_dict[src_type])
            for node_type in x_dict.keys():
                h_dict[node_type] = torch.sum(torch.stack(out_dict[node_type],dim=0), dim=0)
        target_logits = self.out_lin(h_dict[self.target_node_type])
        if get_embeddings:
            h_dict = {node_type:h for node_type,h in h_dict.items()}
            return target_logits, h_dict
        else:
            return target_logits


import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import CiteseerGraphDataset
from sklearn.decomposition import PCA
from dgl.nn import HeteroGraphConv, GraphConv

class HeteroGCNCiteer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, rel_names):
        super().__init__()
        # conv1: in_dim -> hidden_dim for each relation
        self.conv1 = HeteroGraphConv({
            rel: GraphConv(in_dim, hidden_dim)
            for rel in rel_names
        }, aggregate='sum')
        # conv2: hidden_dim -> out_dim for each relation
        self.conv2 = HeteroGraphConv({
            rel: GraphConv(hidden_dim, out_dim)
            for rel in rel_names
        }, aggregate='sum')

    def forward(self, graph, x_dict):
        # x_dict: {'paper': feature_tensor}
        h_dict = self.conv1(graph, x_dict)
        # apply activation
        h_dict = {ntype: F.relu(h) for ntype, h in h_dict.items()}
        h_dict = self.conv2(graph, h_dict)
        return h_dict