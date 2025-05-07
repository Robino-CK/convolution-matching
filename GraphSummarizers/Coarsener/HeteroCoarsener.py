import copy
import dgl
import numpy as np
import os
import sys
from collections import defaultdict

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
from Datasets.NodeClassification.AIFB import AIFB
import pickle
from Datasets.NodeClassification.TestHetero import TestHeteroSmall, TestHeteroBig
import dgl
import torch
from scipy.sparse import coo_matrix
import scipy
from collections import Counter
from copy import deepcopy
CHECKPOINTS = [0.7, 0.5, 0.3, 0.1, 0.05, 0.01, 0.001]

class HeteroCoarsener(GraphSummarizer):
    def __init__(self, dataset: Dataset, original_graph: dgl.DGLGraph, r: float, pairs_per_level: int = 10, 
                 num_nearest_neighbors: int = 10, num_nearest_per_etype:int = 10, filename = "dblp"
                 ):
        """
        A graph summarizer that greedily merges neighboring nodes to summarize a graph that yields
        approximately the same graph convolution.
        :param original_graph: The original DGLGraph instance.
        :param r: The ratio of nodes in the summarized and original graph. Must be in (0, 1].
        :param pairs_per_level: The number of node pairs to merge at each level of coarsening.
        :param  num_nearest_neighbors: ----
        :param num_nearest_per_etype: The number of nearest neighbors to consider for each node per edge type in init step, from here we can get the number of number of edges in merge graph which is in [num_nearest_per_etype, number of etypes * num_nearest_per_etype]  
        
        """
        self.filename = filename
        assert (r > 0.0) and (r <= 1.0)
        self.r = r
        self.pca_components = 3
        self.original_graph = original_graph
        self.dataset = dataset
        self.num_nearest_neighbors = num_nearest_neighbors
        self.num_nearest_per_etype = num_nearest_per_etype
        self.pairs_per_level = pairs_per_level
        self.coarsened_graph = original_graph.clone()
        self.merge_graph = None
        self.node_types = dict()
        self.candidate_pairs = None
        self.homo_hetero_map = dict()
        self.type_homo_mapping = dict()
        self.H_originals_stacked = dict()
        self.H_originals = dict()
        self.label_list_supernodes = dict()
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
            
   
    
        
    def _create_h_spatial_rgcn(self, g):
        start_time = time.time()
        H = dict()
        
        device = "cpu"
        
        for src_type, etype, dst_type in g.canonical_etypes:
            is_features = "feat" in g.nodes[dst_type].data
            degree = self.node_degrees[etype]["out"] + 1 
            degree_inv_src = 1/torch.sqrt(degree) 
            degree_inv_src = degree_inv_src.to(device)
            degree_inv_src = torch.where(degree_inv_src == float("inf"),  torch.tensor(0.0),degree_inv_src) # why does this not work
           
            degree = self.node_degrees[etype]["in"] + 1 
            degree_inv_dest = 1/torch.sqrt(degree) 
            degree_inv_dest = torch.where(degree_inv_dest == float("inf"), torch.tensor(0.0), degree_inv_dest)
           
            if is_features:
                features = g.nodes[dst_type].data["feat"].to(device)
            #degree_inv_dest = degree_inv_dest.to(device)
            H[etype] = dict()
            all_nodes = g.nodes(src_type)
            #H[etype] = 
            for node in all_nodes:
                neighbors = g.successors(node, etype=(src_type, etype, dst_type))
                if is_features:
                    H[etype][node.item()] = torch.zeros(features.shape[1], device=device)
                else:
                    H[etype][node.item()] = torch.zeros(1, device=device)
                for neigh in neighbors:
                    if is_features:
                        feat = features[neigh]
                    else:
                        feat = torch.tensor(1.0, device=device)
                    #feat = feat.to(device)
                    H[etype][node.item()] += degree_inv_src[node] * degree_inv_dest[neigh] * feat
            self.H_originals_stacked[etype] = torch.stack(list(H[etype].values()))
        self.H_originals = H
        print("_create_h_spatial_rgcn", time.time() - start_time )  
    
    
    
    def _init_merge_graph(self, costs):
       
        start_time = time.time()
        self.merge_graphs =dict()
        for ntype in self.coarsened_graph.ntypes:
            num_nodes = self.coarsened_graph.number_of_nodes(ntype=ntype) 
            num_edges = num_nodes * self.num_nearest_per_etype * (len(self.coarsened_graph.canonical_etypes) + 1)
            self.merge_graphs [ntype] = dgl.graph(([], []), num_nodes=self.coarsened_graph.number_of_nodes(ntype=ntype))
            edge_weight_tensor = torch.empty(num_edges) #torch.empty((1,0)).squeeze()
            edge_tensor = torch.empty((2, num_edges), dtype=torch.int64)
            edge_cnt = 0
            for node_pair, n_costs in costs[ntype].items():
                node_1 = node_pair[0]
                node_2 = node_pair[1]
                    #edge = torch.tensor([[node],[neighbor]])
                edge_tensor[0][edge_cnt] = node_1# torch.cat((edge_tensor, edge), dim=1)
                edge_tensor[1][edge_cnt] = node_2
                edge_weight_tensor[edge_cnt] = n_costs ##torch.cat((edge_weight_tensor, torch.tensor([n_costs])))
            
                edge_cnt += 1
                    
            edge_tensor = edge_tensor[:, :edge_cnt]
            edge_weight_tensor = edge_weight_tensor[:edge_cnt]
            self.merge_graphs[ntype].add_edges(edge_tensor[0], edge_tensor[1])
            self.merge_graphs[ntype].edata["edge_weight"] = torch.tensor(edge_weight_tensor)
            self.merge_graphs[ntype].ndata["node_size"] = torch.ones(num_nodes)
            
        print("_init_merge_graph", time.time() - start_time)

    
    def _init_calculate_clostest_edges(self, src_type, etype, dst_type):
        if "feat" in self.coarsened_graph.nodes[dst_type].data:
            H = torch.zeros(( len(self.coarsened_graph.nodes(src_type)),  self.coarsened_graph.nodes[dst_type].data["feat"].shape[1]))
        else:
            H = torch.zeros(( len(self.coarsened_graph.nodes(src_type)), 1))
        for node in self.coarsened_graph.nodes(src_type):
            
            H[node.item(),:] = self.H_originals[etype][node.item()]
        if H.shape[1] > self.pca_components:
            pca = PCA(n_components=self.pca_components)
            H = pca.fit_transform(
            (H - H.mean(dim=0)) / (H.std(dim=0) + 0.0001)
            )   
        kd_tree = scipy.spatial.KDTree(H)
        k = min (self.num_nearest_per_etype, H.shape[0])
        distances, nearest_neighbors =  kd_tree.query(H, k=k, p=1, eps=0.01, workers=1) # TODO
        return (distances, nearest_neighbors)
    
    
    def _init_calculate_clostest_features(self, ntype):
        if "feat" in self.coarsened_graph.nodes[ntype].data:
            H = torch.zeros(( len(self.coarsened_graph.nodes(ntype)),  self.coarsened_graph.nodes[ntype].data["feat"].shape[1]))
        else:
            H = torch.zeros(( len(self.coarsened_graph.nodes(ntype)), 1))
        for node in self.coarsened_graph.nodes(ntype):
            
            H[node.item(),:] = self.coarsened_graph.nodes[ntype].data["feat"][node.item()]
        if H.shape[1] > self.pca_components:
            pca = PCA(n_components=self.pca_components)
            H = pca.fit_transform(
            (H - H.mean(dim=0)) / (H.std(dim=0) + 0.0001)
            )   
        kd_tree = scipy.spatial.KDTree(H)
        k = min (self.num_nearest_per_etype, H.shape[0])
        distances, nearest_neighbors =  kd_tree.query(H, k=k, p=1, eps=0.01, workers=1) # TODO
        return (distances, nearest_neighbors)        
            
    
    def _get_union(self, init_costs):
        start_time = time.time()
        closest_over_all_etypes = dict()
        for src_type, etype, dst_type in self.coarsened_graph.canonical_etypes:
            if not src_type in closest_over_all_etypes:
                closest_over_all_etypes[src_type] = dict()
            for node in self.coarsened_graph.nodes(src_type):
                nearest_neighbor = list(init_costs[etype][1][node.item()])
                if node.item() in nearest_neighbor:
                    nearest_neighbor.remove(node.item())
                if not node.item() in closest_over_all_etypes[src_type]:
                    closest_over_all_etypes[src_type][node.item()] = nearest_neighbor
                else:
                    
                    closest_over_all_etypes[src_type][node.item()] = set(closest_over_all_etypes[src_type][node.item()]).union(nearest_neighbor)
        for ntype in self.coarsened_graph.ntypes:
            for node in self.coarsened_graph.nodes(ntype):
                if node.item() not in closest_over_all_etypes[ntype]:
                    closest_over_all_etypes[ntype][node.item()] = list(init_costs[ntype][1][node.item()])
                else:
                    closest_over_all_etypes[ntype][node.item()] = set(closest_over_all_etypes[ntype][node.item()]).union(list(init_costs[ntype][1][node.item()]))
        print("_get_union", time.time() - start_time)
        return closest_over_all_etypes

    
    def _init_costs_rgcn(self):
        start_time = time.time()
        init_costs = dict()
        
        
        
        for src_type, etype, dst_type in self.coarsened_graph.canonical_etypes:
           init_costs[etype] = self._init_calculate_clostest_edges(src_type, etype, dst_type)
        
        for ntype in self.coarsened_graph.ntypes:
            if "feat" in self.coarsened_graph.nodes[dst_type].data:
                
                init_costs[ntype] = self._init_calculate_clostest_features(ntype)    
              
        
        print("stop init costs", time.time() - start_time)
        return init_costs

    def _find_lowest_cost_edges(self):
        start_time = time.time()
        topk_non_overlapping_per_type = dict()
        for ntype in self.coarsened_graph.ntypes:
            if ntype not in self.merge_graphs:
                continue
            costs = self.merge_graphs[ntype].edata["edge_weight"]
            edges = self.merge_graphs[ntype].edges()
            k = min(self.num_nearest_neighbors * self.pairs_per_level, costs.shape[0]) # TODO 
            lowest_costs = torch.topk(costs, k,largest=False, sorted=True)    
            topk_non_overlapping = list()
            nodes = set()
            # todo
            for edge_index in lowest_costs.indices:
                if len(nodes) > self.pairs_per_level: 
                    break
                src_node = edges[0][edge_index].item()
                dst_node = edges[1][edge_index].item()
                if src_node in nodes or dst_node in nodes:
                    continue
                
                topk_non_overlapping.append((src_node, dst_node))
                nodes.add(src_node)
                nodes.add(dst_node)
            topk_non_overlapping_per_type[ntype] = topk_non_overlapping
        print("_find_lowest_cost_edges", time.time() - start_time)
        return topk_non_overlapping_per_type

    def _sum_edge_costs_over_etypes(self, init_costs, intersection):
        start_time = time.time()
        total_costs = dict()
        for src_type, etype, dst_type in self.coarsened_graph.canonical_etypes:
            if not src_type in total_costs:
                total_costs[src_type] = list()
                for node in self.coarsened_graph.nodes(src_type):
                    
                    nodes = list(init_costs[etype][1][node.item()])
                    for merge_node in (nodes):
                        if merge_node == node or merge_node not in intersection[src_type][node.item()]:
                            continue
                        
                        total_costs[src_type].append((node.item(), merge_node))
        print("stop sum edge costs", time.time() - start_time)
        return total_costs
    
    

    
    def _merge_nodes(self, g, node_type, node_pairs):
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
        start_time = time.time()
        g = deepcopy(g)
        mapping = torch.arange(0, g.num_nodes(ntype=node_type) )
        old_label_list = self.label_list_supernodes.copy()
        
        for node1, node2 in node_pairs: #tqdm(, "merge nodes"):
            g.add_nodes(1, ntype=node_type)
            new_node_id =  g.num_nodes(ntype=node_type) -1
            
            for src_type, etype,dst_type in g.canonical_etypes:
                if src_type != node_type:
                    continue
                edges_original = g.edges(etype=etype)
                mask_node1 =  torch.where(edges_original[0] == mapping[node1], True, False)
                mask_node2 =  torch.where(edges_original[0] == mapping[node2], True, False)
                mask = torch.logical_or(mask_node1, mask_node2)
                edges_dst = torch.unique(edges_original[1][mask])
                edges_src = torch.full(edges_dst.shape,new_node_id )
                g.add_edges(edges_src, edges_dst, etype=(src_type, etype,dst_type))
            
            
            if "feat" in g.nodes[node_type].data:
                old_feats = g.nodes[node_type].data["feat"] 
                g.nodes[node_type].data["feat"][new_node_id] = (old_feats[mapping[node1]] + old_feats[mapping[node2]]) / 2
            # if "label" in g.nodes[node_type].data:
            #     g.nodes[node_type].data["label"][new_node_id] = g.nodes[node_type].data["label"][mapping[node1]] 
                
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
            g.remove_nodes([pre_node1, pre_node2], ntype=node_type) 
        
         
                
        print("_merge_nodes", time.time()- start_time)       
        return g, mapping     
   
   
    def _update_merge_graph_nodes_edges(self, g, node_pairs):
        mapping = torch.arange(0, g.num_nodes() )
        nodes_need_edge_weight_recalc = set()
        for node1, node2 in node_pairs: #tqdm(, "merge nodes"):
            g.add_nodes(1)
            new_node_id =  g.num_nodes() -1
            
            edges_original = g.edges()
            mask_node1 =  torch.where(edges_original[0] == mapping[node1], True, False)
            mask_node2 =  torch.where(edges_original[0] == mapping[node2], True, False)
            mask = torch.logical_or(mask_node1, mask_node2)
            edges_dst = torch.unique(edges_original[1][mask])
            edges_src = torch.full(edges_dst.shape,new_node_id )
            g.add_edges(edges_src, edges_dst)
            
            
            pre_node1 = mapping[node1].item()
            pre_node2 = mapping[node2].item()
            size_node1 = g.ndata["node_size"][pre_node1]
            size_node2 = g.ndata["node_size"][pre_node2]
            g.ndata["node_size"][new_node_id] = size_node1 + size_node2
            mapping[node1] = new_node_id  
            mapping[node2] = new_node_id
                
            if node1 > node2:
                mapping = torch.where(mapping >pre_node1, mapping -1, mapping)
                mapping = torch.where(mapping > pre_node2, mapping -1, mapping)
            else:
                mapping = torch.where(mapping > pre_node2, mapping -1, mapping)
                mapping = torch.where(mapping > pre_node1, mapping -1, mapping)
            nodes_need_edge_weight_recalc.add(node1)
            nodes_need_edge_weight_recalc.add(node2)
            
            g.remove_nodes([pre_node1, pre_node2]) 
        return g, mapping, nodes_need_edge_weight_recalc


    def _update_merge_graph_edge_weigths_features(self, g, ntype, candidates ):
        feat = self.coarsened_graph.nodes[ntype].data['feat']
        device = feat.device

        # 1) Build flat src/dst lists
        keys = list(candidates.keys())
        counts = torch.tensor([len(candidates[k]) for k in keys],dtype=torch.int64, device=device)
        # repeat each node1 for its number of candidates
        src = torch.tensor(keys, device=device, dtype=torch.int64).repeat_interleave(counts)       # shape [E]
        # flatten all node2 lists
        dst = torch.cat([torch.tensor(candidates[k], device=device, dtype=torch.int64) for k in keys], dim=0)  # [E]

        # 2) mask out self‐pairs (if any)
        mask = src != dst
        src, dst = src[mask], dst[mask]                                         # [E']

        # 3) pull features and compute costs
        f1 = feat[src]       # [E'×H]
        f2 = feat[dst]       # [E'×H]
        mid = (f1 - f2) / 2   # [E'×H]
        costs = torch.norm(mid - f1, p=1, dim=1) + torch.norm(mid - f2, p=1, dim=1)  # [E']

        # 4) lookup edge IDs & assign all at once
        eids = g.edge_ids(src, dst)               # [E']
        
        g.edata['edge_weight'][eids] = costs
        
        
   

    def _update_merge_graph_edge_weights_H(self,g, ntype, candidates):
        
       
        device = "cpu"
        
        for src_type, etype, dst_type in self.coarsened_graph.canonical_etypes:
            if src_type != ntype:
                continue
            H_merged = self._create_h_via_cache_vec(candidates, etype, self.merge_graphs[ntype].ndata["node_size"])
            
            
            
            # 1) Build two flat tensors src_nodes, dst_nodes of shape [E], 
            #    where E = total number of (node1→node2) pairs.
            counts = [len(cands) for cands in candidates.values()]
            src_nodes = (
                torch.tensor(list(candidates.keys()), device=device)
                .repeat_interleave(torch.tensor(counts, device=device))
            )                                         # [E]
            dst_nodes = torch.cat(
                [torch.tensor(cands, device=device) for cands in candidates.values()]
            )                                         # [E]

            # 2) Build merged_repr as a big [E×H] tensor by concatenating each H_merged[node1].
            merged_repr = torch.cat([H_merged[n] for n in candidates.keys()], dim=0)  
            #                   → [sum(counts) × hidden_dim] == [E×H]

            # 3) Lookup original reprs for every src and dst in one go:
            src_repr = self.H_originals_stacked[etype][src_nodes]   # [E×H]
            dst_repr = self.H_originals_stacked[etype][dst_nodes]   # [E×H]

            # 4) L1 distances and total cost:
            cost_src = torch.norm(src_repr - merged_repr, p=1, dim=1)  # [E]
            cost_dst = torch.norm(dst_repr - merged_repr, p=1, dim=1)  # [E]
            total_cost = cost_src + cost_dst                          # [E]

            # 5) Fetch all edge IDs and write back in one shot:
            edge_ids = g.edge_ids(src_nodes, dst_nodes)               # [E]
            g.edata['edge_weight'][edge_ids] = total_cost
            
     

    def _update_merge_graph(self, go,  node_pairs, ntype):
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
        start_time = time.time()
        
        g = deepcopy(go)
        
        g, mapping , nodes_need_edge_weight_recalc = self._update_merge_graph_nodes_edges(g, node_pairs)
        candidates_to_update = dict()
        # TODO: take neighboring nodes edges to other nodes into account (neighbors of neighbors) 
        for node in nodes_need_edge_weight_recalc:
            node1 = mapping[node].item()
            succ = g.successors(node1)
            if len(succ) == 0:
                continue
            candidates_to_update[node1] = list()
            for node2 in succ:
                candidates_to_update[node1].append(node2.item())
                
        if len(candidates_to_update) > 0:
            
            self._update_merge_graph_edge_weigths_features(g,ntype, candidates_to_update)
            self._update_merge_graph_edge_weights_H(g, ntype, candidates_to_update)
            print("_update_merge_graph", time.time()- start_time)    
            return g, mapping,  True
        else:
            print("_update_merge_graph: WARNING no more merge candidates", time.time()- start_time)    
            return g, mapping, False
           
            
   
    
    
            
    def _create_h_spatial_via_cache_for_node(self, g_orig, node, conical_type, node2):
        cache = self.H_originals
        s_u = cache[conical_type[1]][node]  
        s_v = cache[conical_type[1]][node2]
        d_u = self.node_degrees[conical_type[1]]["out"][node] + 1
        d_v = self.node_degrees[conical_type[1]]["out"][node2] + 1
        d_u_v = d_u + d_v 
        # TODO
        #k = len(set(g_orig.successors(node, etype=conical_type)) & set(g_orig.successors(node2, etype=conical_type)))  + 1 
     
        h = ((torch.sqrt(torch.tensor(d_u)) * s_u) + (torch.sqrt(torch.tensor(d_v)) * s_v)) / (torch.sqrt(torch.tensor(d_u_v)))
        return h
    
    def _create_h_via_cache_vec(self,  table, etype, cluster_sizes):
        cache = self.H_originals_stacked
       # lists = [sorted(list(v)) for v in cache["authortopaper"].values()]
        H_merged = dict()
        
       # H_merged = torch.zeros((len(table), 5,cache[etype].shape[1]))
        for node1, merge_nodes in table.items(): # TODO: vectorize
            merge_nodes = torch.tensor(list(merge_nodes))
            smth = cache[etype][merge_nodes] * torch.sqrt(self.node_degrees[etype]["out"][merge_nodes].unsqueeze(dim=1)  + cluster_sizes[merge_nodes].unsqueeze(dim=1))
            h = cache[etype][node1] * torch.sqrt(self.node_degrees[etype]["out"][node1] +cluster_sizes[node1]) + smth
            h = h / torch.sqrt(self.node_degrees[etype]["out"][node1] + self.node_degrees[etype]["out"][(merge_nodes)]).unsqueeze(dim=1)
            H_merged[node1] = h
        
        return H_merged
    
    
    def _feature_costs(self, costs_dict, merge_list):
        for ntype in self.coarsened_graph.ntypes:
            feat = self.coarsened_graph.nodes[ntype].data["feat"]
            costs_dict[ntype] = dict()
            # Prepare list of node pairs to compare
            pairs = [(node1, node2)
                    for node1, merge_candidates in merge_list[ntype].items()
                    for node2 in merge_candidates
                    if node1 != node2]

            if pairs:
                node1_ids, node2_ids = zip(*pairs)
                node1_feats = feat[list(node1_ids)]
                node2_feats = feat[list(node2_ids)]

                new_feats = (node1_feats - node2_feats) / 2
                costs = torch.norm(new_feats - node1_feats, p=1, dim=1) + torch.norm(new_feats - node2_feats, p=1, dim=1)

                for (node1, node2), cost in zip(pairs, costs):
                    costs_dict[ntype][(node1, node2)] = cost
    
    def _neighbor_h_costs(self, costs_dict, merge_list):
        
        for src_type, etype, dst_type in self.coarsened_graph.canonical_etypes:
                
            if not src_type in costs_dict:
                costs_dict[src_type] = dict()
            H_merged = self._create_h_via_cache_vec(merge_list[src_type], etype, torch.ones(self.coarsened_graph.number_of_nodes(src_type)))


            
            for node1, merge_candidates in merge_list[src_type].items(): # TODO: vectorize
                node1_repr = self.H_originals_stacked[etype][node1]                # shape: [hidden_dim]
                node2_indices = torch.tensor(list(merge_candidates), device=node1_repr.device)
                node2_repr = self.H_originals_stacked[etype][node2_indices]       # shape: [num_candidates, hidden_dim]
                merged_repr = H_merged[node1]                                     # shape: [num_candidates, hidden_dim]

                # Compute cost: L1 distance from node1 to merged and node2 to merged
                cost_node1 = torch.norm(node1_repr.unsqueeze(0) - merged_repr, p=1, dim=1)  # [num_candidates]
                cost_node2 = torch.norm(node2_repr - merged_repr, p=1, dim=1)               # [num_candidates]
                total_cost = cost_node1 + cost_node2                                        # [num_candidates]

                for node2, cost in zip(merge_candidates, total_cost):
                    if node1 == node2:
                        continue
                    key = (node1, node2)
                    costs_dict[src_type][key] += cost
                    
    def _costs_of_merges(self, merge_list):
        start_time = time.time()
        
        costs_dict = dict()
        self._feature_costs(costs_dict, merge_list)
        self._neighbor_h_costs(costs_dict, merge_list)    
        
        print("_costs_of_merges", time.time() - start_time)
        return costs_dict 
            
            
            
        
       
    
    def _get_master_mapping(self, mappings, ntype):
        master_mapping = dict()
        
        for node in self.original_graph.nodes(ntype):
            node_id = node.item()
            for mapping in mappings:
                node_id = mapping[node_id].item()
            master_mapping[node.item()] = node_id
        return master_mapping
    
    def _get_labels(self, mapping, ntype):
        
        labels_dict = dict()
        inverse_mapping = dict()
        for ori_node, coar_node in mapping.items():
            if coar_node in inverse_mapping:
                inverse_mapping[coar_node].append(ori_node)
            else:
                inverse_mapping[coar_node] = [ori_node]
        for coar_node, ori_list in inverse_mapping.items():
            label_list = []
            for ori_node in ori_list:
                label_list.append(self.original_graph.nodes[ntype].data["label"][ori_node].item())
            counter = Counter(label_list)
            
            labels_dict[coar_node],_ = counter.most_common()[0]
        
        return labels_dict
                
            
    
    
    def init_step(self):
        file_name = f"results/coarsener_{self.filename}_{self.r}_{self.pairs_per_level}_{self.num_nearest_neighbors}_{self.num_nearest_per_etype}.pkl"
        if os.path.exists(file_name):
            print("Loading coarsener from file")
            self = pickle.load(open(file_name, "rb"))
            return 
        self.mappings = dict()
        for ntype in self.coarsened_graph.ntypes:
            self.mappings[ntype] = list()
        
        self._create_h_spatial_rgcn(self.original_graph)
        init_costs = self._init_costs_rgcn()
        union = self._get_union(init_costs)
        
        self.merge_edges = self._costs_of_merges(union)
        self._init_merge_graph(self.merge_edges)
        self.candidates = self._find_lowest_cost_edges()
   
   
   #     pickle.dump(self, open(file_name, "wb"))
    
    
    def iteration_step(self):
        isNewMerges = False
        for ntype, merge_list in self.candidates.items():
            if (self.original_graph.number_of_nodes(ntype) * self.r >  self.coarsened_graph.number_of_nodes(ntype)):
                continue
            self.coarsened_graph, mapping = self._merge_nodes(self.coarsened_graph, ntype, merge_list)
            self.mappings[ntype].append(mapping)
                
            self.merge_graphs[ntype],_, isNewMergesPerType = self._update_merge_graph(self.merge_graphs[ntype], merge_list, ntype)
            isNewMerges = isNewMerges or isNewMergesPerType
            
            
        self.candidates = self._find_lowest_cost_edges()
        return isNewMerges

    def summarize(self, num_steps=20):
        
        self.init_step()
        for i in range(num_steps):
            self.iteration_step()

        

    def get_coarsend_graph(self):
        return self.coarsened_graph
    
    def get_mapping(self, ntype):
        return self._get_master_mapping(self.mappings[ntype], ntype )
        


# dataset = DBLP() 
# original_graph = dataset.load_graph()

# coarsener = HeteroCoarsener(None,original_graph, 0.5, num_nearest_per_etype=3, num_nearest_neighbors=3,pairs_per_level=30)
# coarsener.init_step()
# for i in range(600):
#     print("--------- step: " , i , "---------" )
#     coarsener.iteration_step()