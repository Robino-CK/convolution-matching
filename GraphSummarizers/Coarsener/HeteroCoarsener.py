import copy
import dgl
import numpy as np
import os
import sys
import time
import torch

from abc import abstractmethod
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "../../"))
from Datasets.Dataset import Dataset
from GraphSummarizers.GraphSummarizer import GraphSummarizer
from GraphSummarizers.GraphSummarizer import SUMMARY_GRAPH_FILENAME
from GraphSummarizers.GraphSummarizer import SUMMARY_STATISTICS_FILENAME
from GraphSummarizers.GraphSummarizer import ORIGINAL_TO_SUPER_NODE_FILENAME
from Datasets.NodeClassification.DBLP import DBLP
from Datasets.NodeClassification.TestHetero import TestHeteroSmall, TestHeteroBig
import dgl
import torch
from scipy.sparse import coo_matrix


CHECKPOINTS = [0.7, 0.5, 0.3, 0.1, 0.05, 0.01, 0.001]

class HeteroCoarsener(GraphSummarizer):
    def __init__(self, dataset: Dataset, original_graph: dgl.DGLGraph, r: float, pairs_per_level: int = 10):
        """
        A graph summarizer that greedily merges neighboring nodes to summarize a graph that yields
        approximately the same graph convolution.
        :param original_graph: The original DGLGraph instance.
        :param r: The ratio of nodes in the summarized and original graph. Must be in (0, 1].
        :param pairs_per_level: The number of node pairs to merge at each level of coarsening.
        """
        assert (r > 0.0) and (r <= 1.0)
        self.r = r
        self.original_graph = original_graph
        self.dataset = dataset
        
        self.pairs_per_level = pairs_per_level
        self.coarsened_graph = original_graph.clone()
        self.merge_graph = None
        self.node_types = dict()
        self.candidate_pairs = None
        self.homo_hetero_map = dict()
    
    def _get_adjacency(self):
        # Assume `hg` is your DGL heterograph
        # Convert to homogeneous graph to get global node ID mapping
        
        homo_g = dgl.to_homogeneous(self.coarsened_graph)
        N_total = homo_g.num_nodes()

        adj_matrices = {}  # Store adjacency per edge type

        self.node_types = dict()
        offset = 0
        node_counts = {ntype: self.coarsened_graph.num_nodes(ntype) for ntype in self.coarsened_graph.ntypes}
        for ntype, count in node_counts.items():
            for i in range(offset, count +offset ):
                self.node_types[i] = ntype
                self.homo_hetero_map[i] = i - offset  
            offset += count

        
        for src_type, etype, dest_type in self.coarsened_graph.canonical_etypes :
            
            
            # Or just use the DGL-provided method to get correct homogeneous IDs:
            src_homo, dst_homo = dgl.to_homogeneous(self.coarsened_graph).edge_subgraph(self.coarsened_graph.edges(etype=etype, form='eid')).edges()
            self.total_number_of_nodes = dgl.to_homogeneous(self.coarsened_graph).number_of_nodes()
            # Build sparse adjacency matrix of shape [N_total, N_total]
            data = torch.ones(len(src_homo))
            adj = coo_matrix((data.numpy(), (src_homo.numpy(), dst_homo.numpy())), shape=(N_total, N_total))
            
                        
            values = adj.data
            indices = np.vstack((adj.row, adj.col))

            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            shape = adj.shape

            adj_torch =  torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
            
            adj_matrices[etype] = adj_torch + torch.transpose(adj_torch, 0, 1)
        return adj_matrices
    
    
    def _rgcn_layer(self,k): 
        # no self loops yet!
        adj = self._get_adjacency()
        H = dict()
        H_total = None
        for etype in self.coarsened_graph.etypes:
            adj_tilde = adj[etype] + torch.eye(adj[etype].shape[0])
            deg_left = torch.diag(torch.tensor(adj_tilde.sum(1)).squeeze())
            
            deg_inverted = torch.pow(deg_left, -0.5)
            deg_inverted[deg_inverted == float("Inf")] = 0
            H[etype] = deg_inverted @ adj_tilde @ deg_inverted
            if H_total is None:
                H_total = H[etype]
            else:
                H_total = H_total + H[etype]
        H_total = torch.pow(H_total, k)
        
        return H_total        


    def _init_candidates(self):
        H_total = self._rgcn_layer(2)
        # Get the top-k pairs of nodes to merge
        top_k_pairs = []
        for i in range(H_total.shape[0]):
            for j in range(i + 1, H_total.shape[1]):
                
                top_k_pairs.append((i, j, torch.norm(H_total[:,i] - H_total[:,j], p=2) ))
        top_k_pairs.sort(key=lambda x: x[2], reverse=True)
        top_k_no_overlap = dict()
       
        used_nodes = set()
        i = self.pairs_per_level
        while i > 0 and len(top_k_pairs) > 0:
            node1, node2, _ = top_k_pairs.pop(0)
            if (node1 not in used_nodes and node2 not in used_nodes) and (self.node_types[node1] == self.node_types[node2]):
                top_k_no_overlap[(node1, node2)] = _
                used_nodes.add(node1)
                used_nodes.add(node2)
                i -= 1
        self.candidate_pairs = top_k_no_overlap
        #top_k_pairs = top_k_pairs[:self.pairs_per_level]
       # return top_k_no_overlap
    def _get_partitioning(self, u,v):
        P = torch.eye(self.total_number_of_nodes)
        P = np.delete(P, self.total_number_of_nodes - 1, axis=1)
        P[u,v] = 1
        if not u == self.total_number_of_nodes - 1:
            P[u,u] = 0
            P[-1, u]  = 1  
        return P
            


    def _merge_nodes_hetero(self, g, node_ids_to_merge, node_type):
        """
        Merges a list of nodes in a heterograph (all from the same node_type).
        
        Args:
            g: dgl.DGLHeteroGraph
            node_ids_to_merge: list or tensor of node IDs (same type)
            node_type: str, the type of nodes to merge

        Returns:
            A new DGLHeteroGraph with merged node and reindexed nodes.
        """
        node_ids_to_merge = torch.tensor(node_ids_to_merge)
        new_graph = copy.deepcopy(g)  # Work on a copy to preserve the original

        # 1. Compute new features (average) and label (first one)
        feat = new_graph.nodes[node_type].data['feat'][node_ids_to_merge]
        new_feat = feat.mean(dim=0, keepdim=True)
        
        has_label = 'label' in new_graph.nodes[node_type].data
        if has_label:
            label = new_graph.nodes[node_type].data['label'][node_ids_to_merge[0]].unsqueeze(0)

        # 2. Get all edges involving any of the nodes to be merged
        new_edges = {}
        for etype in new_graph.canonical_etypes:
            src_type, edge_name, dst_type = etype
            src, dst = new_graph.edges(etype=etype)

            # Replace source
            src_mask = (src_type == node_type) & (src.unsqueeze(1) == node_ids_to_merge).any(dim=1)
            dst_mask = (dst_type == node_type) & (dst.unsqueeze(1) == node_ids_to_merge).any(dim=1)

            src = src.clone()
            dst = dst.clone()
            
            # Replace all node_ids_to_merge with a new placeholder ID: temp max id
            new_id = new_graph.num_nodes(node_type)
            if src_type == node_type:
                src[src_mask] = new_id
            if dst_type == node_type:
                dst[dst_mask] = new_id
            
            new_edges[etype] = (src, dst)

        # 3. Remove the old graph
        # Create a new graph with one extra node for the merged node
        num_nodes_dict = {ntype: new_graph.num_nodes(ntype) for ntype in new_graph.ntypes}
        num_nodes_dict[node_type] += 1  #- len(node_ids_to_merge)  # remove merged nodes, add one
        print(new_edges)
        new_graph = dgl.heterograph(new_edges, num_nodes_dict)
        new_graph = dgl.remove_nodes(new_graph, node_ids_to_merge, ntype=node_type)
        print(new_graph)
        # 4. Set new node features
        for ntype in new_graph.ntypes:
            if not "feat" in g.nodes[ntype].data:
                continue
            n = new_graph.num_nodes(ntype)
            old_feats = g.nodes[ntype].data['feat']
            if ntype == node_type:
                # Exclude merged nodes, then add the new feature
                keep_mask = torch.ones(old_feats.shape[0], dtype=torch.bool)
                keep_mask[node_ids_to_merge] = False
                feats = torch.cat([old_feats[keep_mask], new_feat], dim=0)
            else:
                feats = old_feats
            new_graph.nodes[ntype].data['feat'] = feats

            # Handle labels
            if has_label and ntype == node_type:
                old_labels = g.nodes[ntype].data['label']
                labels = torch.cat([old_labels[keep_mask], label], dim=0)
                new_graph.nodes[ntype].data['label'] = labels

        return new_graph

        
    def _merge_nodes_asd(self, node1, node2):
        # Merge the nodes in the graph
        self.coarsened_graph = dgl.merge_nodes(self.coarsened_graph, [node1, node2])
        P = self._get_partitioning(node1, node2)
        # Update the node mapping
        new_node = self.coarsened_graph.num_nodes() - 1
        self.node_mapping[new_node] = (node1, node2)
        return  P
    
    def _merge(self):
        copy_graph = self.coarsened_graph.clone()
        real_costs = dict()
        H_real = self._rgcn_layer(2)
        for node1, node2 in self.candidate_pairs.keys():
            self.coarsened_graph = copy_graph.clone()
            g = self._merge_nodes_hetero(self.coarsened_graph.clone(), [self.homo_hetero_map[node1], self.homo_hetero_map[node2]], self.node_types[node1])
            self.coarsend_graph = g
            H_coarsen = self._rgcn_layer(2)
            
            P = self._get_partitioning(node1, node2)
            costs = 0
            for ntype in self.coarsened_graph.ntypes:
                X_coarsen = self.coarsened_graph.nodes[ntype].data["feat"]
                X_real = copy_graph.nodes[ntype].data["feat"]
                print(P.T@ H_coarsen  @ X_coarsen  )
                costs = torch.norm(P.T@ H_coarsen  @ X_coarsen - H_real @ X_real, 2)
                print(costs)
                continue

    
    def summarize(self):
        pass
    
        
        
        
        

#dataset = DBLP() 
#original_graph = dataset.load_graph()
test = TestHeteroSmall().load_graph()
test = TestHeteroBig().load_graph()
coarsener = HeteroCoarsener(None, test, 0.5)
coarsener._init_candidates()
coarsener._merge()