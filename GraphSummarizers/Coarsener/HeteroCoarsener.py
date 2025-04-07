import copy
import dgl
import numpy as np
import os
import sys
import time
import torch
import sklearn
from abc import abstractmethod
from tqdm import tqdm
from sklearn.decomposition import PCA
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "../../"))
from Datasets.Dataset import Dataset
from GraphSummarizers.GraphSummarizer import GraphSummarizer
from GraphSummarizers.GraphSummarizer import SUMMARY_GRAPH_FILENAME
from GraphSummarizers.GraphSummarizer import SUMMARY_STATISTICS_FILENAME
from GraphSummarizers.GraphSummarizer import ORIGINAL_TO_SUPER_NODE_FILENAME
from Datasets.NodeClassification.DBLP import DBLP
from tqdm import tqdm
from dgl.dataloading import MultiLayerFullNeighborSampler, DataLoader
from Datasets.NodeClassification.TestHetero import TestHeteroSmall, TestHeteroBig
import dgl
import torch
from scipy.sparse import coo_matrix
import scipy
from copy import deepcopy
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
        self.type_homo_mapping = dict()
        self.init_node_info()
        
    
    
    def init_node_info(self):
        
        self.node_degrees = {}

        for etype in self.coarsened_graph.etypes:
            # Compute out-degrees for source nodes of this edge type
            out_deg = self.coarsened_graph.out_degrees(etype=etype)
            # Compute in-degrees for destination nodes of this edge type
            in_deg = self.coarsened_graph.in_degrees(etype=etype)

            # Store degrees per node per relation
            self.node_degrees[etype] = {
                'out': out_deg,
                'in': in_deg
            }
            
            #self.original_graph.nodes()
        
    
    
    def _rgcn_layer(self,k): 
        # no self loops yet!
        
        adj = self._get_adjacency()
        print("get adj")
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

    def _create_h_spatial_rgcn(self, g):
        H = dict()
        for src_type, etype, dst_type in g.canonical_etypes:
            degree = torch.tensor(self.node_degrees[etype]["out"]) 
            degree_inv_src = 1/torch.sqrt(degree)
            degree = torch.tensor(self.node_degrees[etype]["in"]) 
            degree_inv_dest = 1/torch.sqrt(degree)
            H[etype] = dict()
            for node in g.nodes(src_type):
                neighbors = g.successors(node, etype=(src_type, etype, dst_type))
                H[etype][node.item()] = torch.zeros(g.nodes[dst_type].data["feat"].shape[1])
                for neigh in neighbors:
                    H[etype][node.item()] += degree_inv_src[node] * degree_inv_dest[neigh] * g.nodes[dst_type].data["feat"][neigh]
        return H
    
    
    
    def _get_rgcn_edges(self, H_normal):
        init_costs = dict()
        self.nearest_neighbors_keep_rate = 0.1
        self.top_k_nn = 3
        self.num_nearest_neighbors = 3
        print("start init costs")
        for src_type, etype, dst_type in self.coarsened_graph.canonical_etypes:
            if not src_type in init_costs:
                init_costs[src_type] = dict()    
            H = torch.zeros(( len(self.coarsened_graph.nodes(src_type)),  self.coarsened_graph.nodes[dst_type].data["feat"].shape[1]))
            for node in self.coarsened_graph.nodes(src_type):
                
                H[node.item(),:] = H_normal[etype][node.item()]
            if H.shape[1] > 3:
                pca = PCA(n_components=3)
                H = pca.fit_transform(
                (H - H.mean(dim=0)) / (H.std(dim=0) + 0.0001)
                )   
            kd_tree = scipy.spatial.KDTree(H)
            distances, nearest_neighbors =  kd_tree.query(H, k=self.num_nearest_neighbors, p=1, eps=0.01, workers=1)
            init_costs[etype] = (distances, nearest_neighbors)
            continue
            distances = np.array([dist[1:] for dist in distances])
            
            print("RGCN Distances Min:{}, Max:{}, {} Percentile:{}, Mean:{}".format(
                distances.min(), distances.max(), self.nearest_neighbors_keep_rate,
                np.percentile(distances, self.nearest_neighbors_keep_rate * 100), distances.mean()))

            top_k_neighbor_pairs = np.array([(i, nearest_neighbors[i][j]) for i in range(nearest_neighbors.shape[0])
                                            for j in range(1, self.top_k_nn + 1)])
            merge_graph_edges = torch.tensor(top_k_neighbor_pairs)

            all_neighbor_pairs = np.array([(i, nearest_neighbors[i][j]) for i in range(nearest_neighbors.shape[0])
                                        for j in range(1, len(nearest_neighbors[i]))])
            distances = distances.flatten()
            nearest_node_indices = np.argsort(distances)[:int(len(distances) * self.nearest_neighbors_keep_rate)]
            merge_graph_edges = torch.cat([merge_graph_edges, torch.tensor(all_neighbor_pairs[nearest_node_indices])])
            
            #exact_neighbors = kd_tree.query_ball_point(H,
            #                                       r=0.01, p=2, workers=1)
            
            #exact_neighbor_pairs = np.array([(i, exact_neighbors[i][j]) for i in range(exact_neighbors.shape[0])
            #                                for j in range(1, len(exact_neighbors[i]))])

            #print("SGCN exact_neighbors Count:{}".format(len(exact_neighbor_pairs)))

            #if len(exact_neighbor_pairs) > 0:
            #    merge_graph_edges = torch.cat([merge_graph_edges, torch.tensor(exact_neighbor_pairs)])

            # Remove duplicates.
            merge_graph_edges = merge_graph_edges.sort(dim=1)[0].unique(dim=0)
            # Remove self loops.
         
            merge_graph_edges= merge_graph_edges[merge_graph_edges[:, 0] != merge_graph_edges[:, 1]]
                
            self.merge_graph = dgl.graph((merge_graph_edges[:, 0], merge_graph_edges[:, 1]),
                                        num_nodes=self.summarized_graph.number_of_nodes())

            self.merge_graph.edata["edge_weight"] = self.summarized_graph.has_edges_between(
                *self.merge_graph.edges(form="uv")).to(dtype=torch.float32)
        return init_costs
    
            
              
    def _find_lowest_cost_edges(self, init_costs):
        closest_over_all_etypes = dict()
        for src_type, etype, dst_type in self.coarsened_graph.canonical_etypes:
            #
            if not src_type in closest_over_all_etypes:
                closest_over_all_etypes[src_type] = dict()
            for node in self.coarsened_graph.nodes(src_type):
                nearest_neighbor = init_costs[etype][1][node.item()]
                if not node.item() in closest_over_all_etypes[src_type]:
                    closest_over_all_etypes[src_type][node.item()] = list(nearest_neighbor)
                else:
                    
                    closest_over_all_etypes[src_type][node.item()] = set(closest_over_all_etypes[src_type][node.item()]).intersection(list(nearest_neighbor))
    
        return closest_over_all_etypes

    
    def _select_candidates(self, closest_over_all_etypes):
        merge_list = dict()
        already_merged = dict()
        for src_type, etype, dst_type in self.coarsened_graph.canonical_etypes:
            if not src_type in merge_list:
                merge_list[src_type] = []
                already_merged[src_type] = set()
            for node, candidates in closest_over_all_etypes[src_type].items():
                if len(candidates) > 1:
                    best_candidate = candidates.pop()
                    if best_candidate == node:
                        continue
                    if not best_candidate in already_merged[src_type] and not node in already_merged[src_type]:
                        merge_list[src_type].append((best_candidate, node))
                        already_merged[src_type].add(best_candidate)
                        already_merged[src_type].add(node)
        return merge_list

            
    def _costs_of_merges(self, merge_list):
        H_original = self._create_h_spatial_rgcn(self.coarsened_graph)
        costs = dict()
        for node_type, merge_candidates in merge_list.items():
            
            for node1, node2 in merge_candidates:
                merged_graph = self._merge_nodes_hetero(self.coarsened_graph,[node1,node2], node_type)
                P = self._get_partitioning(node2, node1, self.coarsened_graph.num_nodes(ntype=node_type))
                H_merged = self._create_h_spatial_rgcn(merged_graph)
                costs = 0
                for src_type,etype,dst_type in self.coarsened_graph.canonical_etypes:
                    
                    H_type_merged = torch.stack(list(H_merged[etype].values()))
                    H_type_orig = torch.stack(list(H_original[etype].values()))
                    if src_type == node_type:
                        costs += torch.norm(P@H_type_merged - H_type_orig, 2)
                    else:
                        costs += torch.norm(H_type_merged - H_type_orig, 2)
                    print(costs)
                    
                    
    def _merge_all_paper(self, merge_list):
        merge_graph = self.coarsened_graph.clone()
        
        #for node_type, merge_candidates in merge_list.items():
        #    merge_list = []
        node_type = "paper"
        merge_candidates = merge_list["paper"]
        merge_graph , _ = self.merge_node_pairs(merge_graph,node_type, merge_candidates)
            
               
        return merge_graph
       
    def _get_partitioning(self, u,v , number_of_nodes):
        P = torch.eye(number_of_nodes)
        P = np.delete(P, number_of_nodes - 1, axis=1)
        P[u,v] = 1
        if not u == number_of_nodes - 1:
            P[u,u] = 0
            P[-1, u]  = 1  
        return P
    
    def _merge_nodes_213(self, g, node_type, node_pairs):
        """
        Merge multiple node pairs in a heterogeneous DGL graph.

        Args:
            g (dgl.DGLHeteroGraph): The input graph.
            node_type (str): The node type of the nodes to merge.
            node_pairs (list of tuple): List of (node1, node2) pairs to merge.
            feat_key (str): The feature key to average.

        Returns:
            new_g (dgl.DGLHeteroGraph): Graph with merged nodes.
            mapping (dict): Mapping from original node IDs to supernode ID.
        """
       
        g = deepcopy(g)
        
        mapping = torch.arange(0, g.num_nodes(ntype=node_type) )
        
        for node1, node2 in node_pairs:
            for src_type, etype,dst_type in g.canonical_etypes:
                if src_type != node_type:
                    continue
                edges_original = g.edges(etype=etype)
                mask_node1 =  torch.where(edges_original[0] == mapping[node1], True, False)
                mask_node2 =  torch.where(edges_original[0] == mapping[node2], True, False)
                mask = torch.logical_or(mask_node1, mask_node2)
                edges_dst = torch.unique(edges_original[1][mask])
                new_node_id =  g.num_nodes(ntype=src_type)
                edges_src = torch.full(edges_dst.shape,new_node_id )
                g.add_edges(edges_src, edges_dst, etype=(src_type, etype,dst_type))
                if "feat" in g.nodes[src_type].data:
                    old_feats = g.nodes[src_type].data["feat"] 
                    g.nodes[src_type].data["feat"][new_node_id] = (old_feats[mapping[node1]] + old_feats[mapping[node2]]) / 2
                if "label" in g.nodes[src_type].data:
                    g.nodes[src_type].data["label"][new_node_id] = g.nodes[src_type].data["label"][mapping[node1]] 
                pre_node1 = mapping[node1].item()
                pre_node2 = mapping[node2].item()
                mapping[node1] = new_node_id  
                mapping[node2] = new_node_id
                
                if node1 > node2:
                    mapping = torch.where(mapping >pre_node1, mapping -1, mapping)
                    mapping = torch.where(mapping > pre_node2, mapping -1, mapping)
                else:
                    mapping = torch.where(mapping > pre_node2, mapping -1, mapping)
                    mapping = torch.where(mapping > pre_node1, mapping -1, mapping)
                
                g.remove_nodes([pre_node1, pre_node2], ntype=src_type)
                
        return g, mapping     
   


    
    def summarize(self):
        pass
    
        
    def _merge_new():
        
        pass
    
    
    

dataset = DBLP() 
original_graph = dataset.load_graph()


test = TestHeteroSmall().load_graph()

test = TestHeteroBig().load_graph()

coarsener = HeteroCoarsener(dataset, original_graph, 0.5)


H = coarsener._create_h_spatial_rgcn(original_graph)
candidates = coarsener._select_candidates(coarsener._find_lowest_cost_edges(coarsener._get_rgcn_edges(H)))
merged_graph = coarsener._merge_nodes_213(original_graph, "author", candidates["author"])
test = TestHeteroSmall().load_graph()

         

coarsener = HeteroCoarsener(None, test, 0.5)
coarsener._merge_nodes_213(test, "author", [(0,2), (1,3)])