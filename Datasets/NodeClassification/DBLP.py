import torch
from dgl import heterograph
import dgl
import torch_geometric.transforms as T
import torch_geometric

import dgl
import os
import sys
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "../../"))


class DBLP():
    def __init__(self):
        pass    
        #super().__init__(name='Cora')

   
   #     self.epochs = 600

    def load_graph(self) -> dgl.DGLGraph:
        
        # Load the dataset
        path = './dblp'
        dataset = torch_geometric.datasets.DBLP(path)
        data = dataset[0]  # Only one graph

        node_types = ['author', 'paper', 'term', 'conference']  #
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
            edge_tuples[(src_type, rel_type, dst_type)] = (src_nodes, dst_nodes)
        # Create DGL heterograph
        g = heterograph(edge_tuples, num_nodes_dict=num_nodes_dict)

        # You can also assign features
        for ntype in g.ntypes:
            if ntype == "conference":
                continue
            #if 'x' in data.x_dict[ntype]:
            g.nodes[ntype].data['feat'] = data.x_dict[ntype]
        
        return g

