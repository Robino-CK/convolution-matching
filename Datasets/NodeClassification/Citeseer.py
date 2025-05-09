import dgl
import os
import sys

import torch.nn.functional

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "../../"))
from Datasets.NodeClassification.NodeClassificationDataset import NodeClassificationDataset

from dgl.data import CiteseerGraphDataset
import torch
from sklearn.decomposition import PCA

class Citeseer(NodeClassificationDataset):
    def __init__(self):
        pass

    def load_graph(self) -> dgl.DGLGraph:
        # Load the homogeneous Citeseer citation graph
        dataset = CiteseerGraphDataset()
        g = dataset[0]

        # Original node features
        data = g.ndata['feat']  # shape: (num_nodes, orig_feat_dim)


        # Reduce to 50 dimensions (you can adjust this)
        reduced_feat = self.reduce_features(data, n_components=50)

        # Build heterogeneous graph with 'cites' and 'cited-by' relations
        src, dst = g.edges()
        data_dict = {
            ('paper', 'cites', 'paper'): (src, dst),
            ('paper', 'cited-by', 'paper'): (dst, src)
        }
        hetero_g = dgl.heterograph(data_dict)

        # Assign reduced features and other data
        hetero_g.nodes['paper'].data['feat'] = reduced_feat
        hetero_g.nodes['paper'].data['label'] = g.ndata['label']
        for key in ['train_mask', 'val_mask', 'test_mask']:
            hetero_g.nodes['paper'].data[key] = g.ndata[key]

        return hetero_g
    
    # Apply PCA to reduce feature dimension
    # Choose the number of principal components
    def reduce_features(self,feat_tensor, n_components=50):
        """
        Reduce feature dimension using PCA.

        Parameters
        ----------
        feat_tensor : torch.Tensor
            Input node features of shape (N, D).
        n_components : int
            Target number of dimensions.

        Returns
        -------
        torch.Tensor
            Transformed features of shape (N, n_components).
        """
        # Convert to numpy for sklearn PCA
        X = feat_tensor.numpy()
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        return torch.from_numpy(X_reduced)
