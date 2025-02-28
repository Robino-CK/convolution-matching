import dgl
import os
import sys
import torch
import torch.nn.functional
import networkx as nx
import matplotlib.pyplot as plt  # Ensure plt is not redefined
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "../../"))
from Datasets.NodeClassification.NodeClassificationDataset import NodeClassificationDataset


class Karate(NodeClassificationDataset):
    def __init__(self):
        super().__init__(name='Karate')

        self.epochs = 6

    def load_graph(self) -> dgl.DGLGraph:
        graph = dgl.data.KarateClubDataset()[0]
        # Add random features to the nodes
        random_features = torch.rand((graph.number_of_nodes(), 10)) * 0.01  # Assuming 10 features per node
        graph.ndata["feat"] = random_features
        graph.ndata["feat"] = torch.nn.functional.normalize(graph.ndata["feat"], p=1.0)
        
        # Create train, validation, and test masks
        num_nodes = graph.number_of_nodes()
        indices = np.arange(num_nodes)
        np.random.shuffle(indices)
        train_size = int(0.8 * num_nodes)
        val_size = int(0.1 * num_nodes)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
        
        graph.ndata['train_mask'] = train_mask
        graph.ndata['val_mask'] = val_mask
        graph.ndata['test_mask'] = test_mask
        
        return graph
    
    
g = Karate().load_graph()

options = {
    'node_color': 'black',
    'node_size': 20,
    'width': 1,
}
G = dgl.to_networkx(g)
plt.figure(figsize=[15, 7])  # Ensure plt is not redefined
nx.draw_networkx(G, **options)
plt.savefig("/home/robin/uu/thesis/convolution-matching/Karate.png")