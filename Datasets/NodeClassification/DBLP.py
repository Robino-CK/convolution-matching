import torch
from dgl import heterograph
import dgl
import torch_geometric.transforms as T
import torch_geometric

import dgl
import os
import sys
import torch
from torch.serialization import add_safe_globals
from torch_geometric.data import HeteroData
from torch_geometric.data.graph_store import EdgeAttr
from torch_geometric.data.storage import NodeStorage
from torch_geometric.data.storage import BaseStorage
from torch_geometric.data.storage import EdgeStorage
import torch
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "../../"))


class DBLP():
    def __init__(self):
        pass    
        #super().__init__(name='Cora')

   
   #     self.epochs = 600

    def load_graph(self) -> dgl.DGLGraph:
        torch.serialization.add_safe_globals([HeteroData])
        torch.serialization.add_safe_globals([HeteroData])
        torch.serialization.add_safe_globals([EdgeAttr])
        torch.serialization.add_safe_globals([torch_geometric.data.TensorAttr])
        torch.serialization.add_safe_globals([BaseStorage])
        torch.serialization.add_safe_globals([NodeStorage])
        torch.serialization.add_safe_globals([EdgeStorage])
        
        # Load the dataset
        path = './dblp'
        dataset = torch_geometric.datasets.DBLP(path)
        data = dataset[0]  # Only one graph

        num_nodes_dict = {
            'author': data.x_dict['author'].shape[0],
            'paper': data.x_dict['paper'].shape[0],
            'term': data.x_dict['term'].shape[0],
            'conference': data["conference"]["num_nodes"]
        }

        # Extract edge index dict from torch geometric
        edge_index_dict = data.edge_index_dict

        # Convert PyG edge index dict to DGL edge tuples
        edge_tuples = {}
        for (src_type, rel_type, dst_type), edge_index in edge_index_dict.items():
            src_nodes = edge_index[0].numpy()
            dst_nodes = edge_index[1].numpy()
            edge_tuples[(src_type, src_type + rel_type + dst_type, dst_type)] = (src_nodes, dst_nodes)
        # Create DGL heterograph
        g = heterograph(edge_tuples, num_nodes_dict=num_nodes_dict)

        # You can also assign features
        for ntype in g.ntypes:
            if ntype == "conference":
                g.nodes[ntype].data['feat'] = torch.ones(num_nodes_dict['conference'],1)    
            #if 'x' in data.x_dict[ntype]:
            else:
                g.nodes[ntype].data['feat'] = data.x_dict[ntype]
        
        return g

