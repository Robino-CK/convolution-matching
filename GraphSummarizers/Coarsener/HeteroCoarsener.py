import copy
import dgl
import numpy as np
import os
import sys
import math
import time
import torch
from sklearn.decomposition import PCA
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "../../"))
from Datasets.Dataset import Dataset

from sklearn.preprocessing import MinMaxScaler
from GraphSummarizers.GraphSummarizer import GraphSummarizer
from GraphSummarizers.GraphSummarizer import SUMMARY_GRAPH_FILENAME
from GraphSummarizers.GraphSummarizer import SUMMARY_STATISTICS_FILENAME
from GraphSummarizers.GraphSummarizer import ORIGINAL_TO_SUPER_NODE_FILENAME
from Datasets.NodeClassification.DBLP import DBLP
from Datasets.NodeClassification.AIFB import AIFB
from Datasets.NodeClassification.Citeseer import Citeseer

import pickle
import dgl
import torch
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
        
        self.label_list_supernodes = dict()
        self.init_node_info()
        
        self.minmax_ntype = dict()
        self.minmax_etype = dict()
        
    
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
        S = dict()
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
            S[etype] = dict()
            all_nodes = g.nodes(src_type)
            #H[etype] =
            
            self.coarsened_graph.nodes[src_type].data[f's{etype}'] = torch.zeros((len(all_nodes), features.shape[1]), device=device) 
            self.coarsened_graph.nodes[src_type].data[f'h{etype}'] = torch.zeros((len(all_nodes), features.shape[1]), device=device)
            self.coarsened_graph.nodes[src_type].data['node_size'] = torch.ones((len(all_nodes), 1), device=device)
            for node in all_nodes:
                neighbors = g.successors(node, etype=(src_type, etype, dst_type))
                if is_features:
                    H[etype][node.item()] = torch.zeros(features.shape[1], device=device)
                    S[etype][node.item()] = torch.zeros(features.shape[1], device=device)
                else:
                    H[etype][node.item()] = torch.zeros(1, device=device)
                    S[etype][node.item()] = torch.zeros(1, device=device)
                for neigh in neighbors:
                    if is_features:
                        feat = features[neigh]
                    else:
                        feat = torch.tensor(1.0, device=device)
                    #feat = feat.to(device)
                    s = degree_inv_dest[neigh] * feat
                    self.coarsened_graph.nodes[src_type].data[f's{etype}'][node] += s
                    self.coarsened_graph.nodes[src_type].data[f'h{etype}'][node] +=  degree_inv_src[node] * s
            
        
        print("_create_h_spatial_rgcn", time.time() - start_time )  
    
    
    
    def _init_merge_graph(self, costs, edges):
       
        start_time = time.time()
        self.merge_graphs =dict()
        for ntype in self.coarsened_graph.ntypes:
            num_nodes = self.coarsened_graph.number_of_nodes(ntype=ntype) 
            self.merge_graphs [ntype] = dgl.graph(([], []), num_nodes=self.coarsened_graph.number_of_nodes(ntype=ntype))
            
            self.merge_graphs[ntype].add_edges(edges[ntype][0,:],  edges[ntype][1,:])
            self.merge_graphs[ntype].edata["edge_weight"] = costs[ntype]
            
        print("_init_merge_graph", time.time() - start_time)

    
    def _init_calculate_clostest_edges(self, src_type, etype, dst_type):
        if "feat" in self.coarsened_graph.nodes[dst_type].data:
            H = torch.zeros(( len(self.coarsened_graph.nodes(src_type)),  self.coarsened_graph.nodes[dst_type].data["feat"].shape[1]))
        else:
            H = torch.zeros(( len(self.coarsened_graph.nodes(src_type)), 1))
        for node in self.coarsened_graph.nodes(src_type):
            
            H[node.item(),:] = self.coarsened_graph.nodes[src_type].data[f'h{etype}'] [node.item()]
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
                if not "feat" in self.coarsened_graph.nodes[dst_type].data:
                    continue
                nearest_neighbor = list(init_costs[ntype][1][node.item()])
                if node.item() in nearest_neighbor:
                    nearest_neighbor.remove(node.item())
                if node.item() not in closest_over_all_etypes[ntype]:
                    closest_over_all_etypes[ntype][node.item()] = nearest_neighbor
                else:
                    closest_over_all_etypes[ntype][node.item()] = set(closest_over_all_etypes[ntype][node.item()]).union(nearest_neighbor)
        
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
        
        for node1, node2 in node_pairs: #tqdm(, "merge nodes"):
            g.add_nodes(1, ntype=node_type)
            new_node_id =  g.num_nodes(ntype=node_type) -1
            
            for src_type, etype,dst_type in g.canonical_etypes:
                if src_type != node_type:
                    continue
                edges_original = g.edges(etype=etype)
                mask_node1 =  edges_original[0] == (mapping[node1]).item()
                mask_node2 =  edges_original[0] == mapping[node2].item()
                mask = torch.logical_or(mask_node1, mask_node2)
                edges_dst = torch.unique(edges_original[1][mask])
                edges_src = torch.full(edges_dst.shape,new_node_id )
                g.add_edges(edges_src, edges_dst, etype=(src_type, etype,dst_type))
                
                mask_node1 =  edges_original[1] == (mapping[node1]).item()
                mask_node2 =  edges_original[1] == mapping[node2].item()
                mask = torch.logical_or(mask_node1, mask_node2)
                edges_dst = torch.unique(edges_original[0][mask])
                edges_src = torch.full(edges_dst.shape,new_node_id )
                g.add_edges(edges_dst, edges_src, etype=(src_type, etype,dst_type))
                
                
                
                suv = g.nodes[node_type].data[f's{etype}'][mapping[node1]] + g.nodes[node_type].data[f's{etype}'][mapping[node2]]
                cuv = g.nodes[node_type].data["node_size"][mapping[node1]] + g.nodes[node_type].data["node_size"][mapping[node2]]
                g.nodes[node_type].data["node_size"][new_node_id] = cuv
                duv = g.out_degrees(new_node_id, etype=etype) # TODO
                g.nodes[node_type].data[f's{etype}'][new_node_id] = suv  
                g.nodes[node_type].data[f'h{etype}'][new_node_id] =  suv / np.sqrt(duv + cuv)
            
            
            if "feat" in g.nodes[node_type].data:
                old_feats = g.nodes[node_type].data["feat"] 
                g.nodes[node_type].data["feat"][new_node_id] = (old_feats[mapping[node1]] + old_feats[mapping[node2]]) / 2
            
                
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
        start_time = time.time()
        mapping = torch.arange(0, g.num_nodes() )
        nodes_need_edge_weight_recalc = set()
        
        g.edata["needs_check"] = torch.zeros(g.num_edges(), dtype=torch.bool) 
        for node1, node2 in node_pairs: #tqdm(, "merge nodes"):
            g.add_nodes(1)
            new_node_id =  g.num_nodes() -1
            
            edges_original = g.edges()
            mask_node1 =  torch.where(edges_original[0] == mapping[node1], True, False)
            mask_node2 =  torch.where(edges_original[0] == mapping[node2], True, False)
            mask = torch.logical_or(mask_node1, mask_node2)
            # TODO: dst nneds to be checked maybe too
            edges_dst = torch.unique(edges_original[1][mask])
            edges_src = torch.full(edges_dst.shape,new_node_id )
            g.add_edges(edges_src, edges_dst)
            edge_ids = g.edge_ids(edges_src, edges_dst)
            g.edata["needs_check"][edge_ids] = True
            
            mask_node1 =  torch.where(edges_original[1] == mapping[node1], True, False)
            mask_node2 =  torch.where(edges_original[1] == mapping[node2], True, False)
            mask = torch.logical_or(mask_node1, mask_node2)
            edges_dst = torch.unique(edges_original[0][mask])
            edges_src = torch.full(edges_dst.shape,new_node_id )
            g.add_edges(edges_dst , edges_src)
            edge_ids = g.edge_ids(edges_dst, edges_src)
            g.edata["needs_check"][edge_ids] = True
            
            
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
            nodes_need_edge_weight_recalc.add(new_node_id)
            nodes_need_edge_weight_recalc.add(node2)
            
            g.remove_nodes([pre_node1, pre_node2]) 
        print("_update_merge_graph_nodes_edges", time.time()- start_time)   
        return g, mapping


    def _update_merge_graph_edge_weigths_features(self, g, ntype, eids ):
        start_time = time.time()
        if "feat" not in self.coarsened_graph.nodes[ntype].data:
            return
        feat = self.coarsened_graph.nodes[ntype].data['feat']
        device = feat.device

        # 1) Build flat src/dst lists
        src , dst = g.find_edges(eids)  
      
        # 2) mask out self‐pairs (if any)
        mask = src != dst
        src, dst = src[mask], dst[mask]                                         # [E']

        # 3) pull features and compute costs
        f1 = feat[src]       # [E'×H]
        f2 = feat[dst]       # [E'×H]
        mid = (f1 + f2) / 2   # [E'×H]
        costs = (torch.norm(mid - f1,  dim=1, p =1) + torch.norm(mid - f2,  dim=1, p=1))  / (self.minmax_ntype[ntype][2])  # [E']
        print("_update_merge_graph_edge_weigths_features", time.time()- start_time)
        return  costs
        
        
   

    def _update_merge_graph_edge_weights_H(self,g, ntype, eids):
        
        start_time = time.time()
        device = "cpu"
        
        src_nodes , dst_nodes = g.find_edges(eids)  
        candidates = dict()
        
        for src_n, dst_n in zip(src_nodes, dst_nodes):
            if not src_n.item() in candidates:
                candidates[src_n.item()] = set()
            candidates[src_n.item()].add(dst_n.item())
        print("candidates time" , time.time()- start_time)
        start_time = time.time()   
        costs = torch.zeros(len(src_nodes), device=device)
        for src_type, etype, dst_type in self.coarsened_graph.canonical_etypes:
            if src_type != ntype:
                continue
            
            
            H_merged = self._create_h_via_cache_vec(candidates,ntype, etype,  self.coarsened_graph.nodes[src_type].data['node_size'])
            
            
            merged_repr = torch.cat([H_merged[n] for n in candidates.keys()], dim=0)  

            src_repr = self.coarsened_graph.nodes[src_type].data[f'h{etype}'][src_nodes]   # TODO: wrong !!!
            dst_repr = self.coarsened_graph.nodes[src_type].data[f'h{etype}'][dst_nodes]   

            cost_src =torch.norm(src_repr - merged_repr,  dim=1, p=1)  # [E]
            cost_dst = torch.norm(dst_repr - merged_repr,  dim=1, p=1)  # [E]
            total_cost = (cost_src + cost_dst) / (self.minmax_etype[etype][2])               # [E]
            costs += total_cost            
            
        print("_update_merge_graph_edge_weights_H", time.time()- start_time)
        return  costs   
         

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
        
        g, mapping = self._update_merge_graph_nodes_edges(g, node_pairs)
        candidates_to_update = dict()
        # TODO: take neighboring nodes edges to other nodes into account (neighbors of neighbors) 
        self.edge_ids_need_recalc = g.edata["needs_check"].nonzero(as_tuple=True)[0]
                
        if len(self.edge_ids_need_recalc) > 0:    
            self.costs_features = self._update_merge_graph_edge_weigths_features(g,ntype, self.edge_ids_need_recalc)
            self.costs_H = self._update_merge_graph_edge_weights_H(g, ntype, self.edge_ids_need_recalc)
            
            g.edata["edge_weight"][self.edge_ids_need_recalc] = self.costs_features 
            g.edata["edge_weight"][self.edge_ids_need_recalc] += self.costs_H
            print("_update_merge_graph", time.time()- start_time)    
            return g, mapping,  True
        else:
            print("_update_merge_graph: WARNING no more merge candidates", time.time()- start_time)    
            return g, mapping, False
           
            
   
    

    
    def _create_h_via_cache_vec(self,  table,ntype, etype, cluster_sizes):
        cache = self.coarsened_graph.nodes[ntype].data[f's{etype}']
       # lists = [sorted(list(v)) for v in cache["authortopaper"].values()]
        H_merged = dict()
        
       # H_merged = torch.zeros((len(table), 5,cache[etype].shape[1]))
        for node1, merge_nodes in table.items(): # TODO: vectorize
            merge_nodes = torch.tensor(list(merge_nodes))
            degree1 = self.coarsened_graph.out_degrees(etype=etype)[node1]
            degree2 = self.coarsened_graph.out_degrees(etype=etype)[merge_nodes]
            # merge_s = (cache[merge_nodes] * torch.sqrt(degree2  + cluster_sizes[merge_nodes].squeeze()).unsqueeze(1) )
            # h = cache[node1] * torch.sqrt(degree1 +cluster_sizes[node1]) + merge_s
            
            su = cache[node1]
            sv = cache[merge_nodes]
            su = su.repeat(merge_nodes.shape[0], 1)
            duv = degree1 + degree2 # TODO depended on if nodes are connected
            cuv = cluster_sizes[node1] + cluster_sizes[merge_nodes]
            
            h = (su + sv) / torch.sqrt(duv + cuv.squeeze()).unsqueeze(1)
            H_merged[node1] = h
        
        return H_merged
    
    
    def _feature_costs(self,  merge_list):
        costs_dict = dict()
        for ntype in self.coarsened_graph.ntypes:
            if "feat" not in self.coarsened_graph.nodes[ntype].data:
                continue
            feat = self.coarsened_graph.nodes[ntype].data["feat"]
            #dim_normalization = np.sqrt(feat.shape[1])
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
                cost_array = torch.zeros(len(pairs))
                index_array = torch.zeros(2,len(pairs), dtype=torch.int64)

                new_feats = (node1_feats + node2_feats) / 2
                costs = torch.norm(new_feats - node1_feats,  dim=1,p=1) + torch.norm(new_feats - node2_feats,  dim=1, p=1)
                index = 0
                for (node1, node2), cost in zip(pairs, costs):
                    cost_array[index] = cost
                    index_array[0][index] = node1
                    index_array[1][index] = node2
                    index += 1
                costs_dict[ntype]["costs"] = cost_array
                costs_dict[ntype]["index"] = index_array
        return costs_dict
    
    def _h_costs(self,  merge_list):
        costs_dict = dict()
        for src_type, etype, dst_type in self.coarsened_graph.canonical_etypes:
                
            if not src_type in costs_dict:
                costs_dict[src_type] = dict()
            if not etype in costs_dict[src_type]:
                costs_dict[src_type][etype] = dict()
            
            
            
            H_merged = self._create_h_via_cache_vec(merge_list[src_type], src_type, etype, torch.ones(self.coarsened_graph.number_of_nodes(src_type)))
            costs_array = torch.zeros(len(merge_list[src_type]) * self.num_nearest_per_etype *  len(self.coarsened_graph.canonical_etypes) * 10 )
            index_array = torch.zeros(2,len(merge_list[src_type])  *  self.num_nearest_per_etype *(len(self.coarsened_graph.canonical_etypes)  * 10), dtype=torch.int64)

                
            index = 0
            for node1, merge_candidates in merge_list[src_type].items(): # TODO: vectorize
                node1_repr =   self.coarsened_graph.nodes[src_type].data[f'h{etype}'][node1]              # shape: [hidden_dim]
                node2_indices = torch.tensor(list(merge_candidates), device=node1_repr.device)
                node2_repr = self.coarsened_graph.nodes[src_type].data[f'h{etype}'][node2_indices]       # shape: [num_candidates, hidden_dim]
                merged_repr = H_merged[node1]                                     # shape: [num_candidates, hidden_dim]

                # Compute cost: L1 distance from node1 to merged and node2 to merged
                cost_node1 =torch.norm(node1_repr.unsqueeze(0)  - merged_repr,  dim=1, p=1)  # [num_candidates]
                cost_node2 = torch.norm(node2_repr - merged_repr, dim=1, p=1)              # [num_candidates]
                total_cost = cost_node1 + cost_node2                                        # [num_candidates]

                for node2, cost in zip(merge_candidates, total_cost):
                    if node1 == node2:
                        continue
                    costs_array[index] = cost
                    index_array[0][index] = node1
                    index_array[1][index] = node2
                    index += 1
            costs_dict[src_type][etype]["costs"] = costs_array[:index]
            costs_dict[src_type][etype]["index"] = index_array[:, :index]
        return costs_dict 

    def _neighbors_h_costs(self, merge_list):
        
        pass
    
    def _add_costs(self, costs_dict_feat, costs_dict_etype):
        start_time = time.time()
        costs_dict = dict()
        index_dict = dict()
        for ntype in costs_dict_feat:
            
            costs_array = costs_dict_feat[ntype]["costs"]
            index_array = costs_dict_feat[ntype]["index"]
            
            minimum = torch.min(costs_array)
            maximum = torch.max(costs_array)
            R = (maximum - minimum + 0.0000000000000001)
            self.minmax_ntype[ntype] = (minimum, maximum,R)
            costs_array = (costs_array / R)
            costs_dict[ntype] = costs_array
            index_dict[ntype] = index_array
                
                
        for src_type,ecosts in costs_dict_etype.items():
            for etype, node_pair_costs in ecosts.items():
                costs_array = node_pair_costs["costs"]
                index_array = node_pair_costs["index"]
                minimum = torch.min(costs_array)
                maximum = torch.max(costs_array)
                R = (maximum - minimum + 0.0000000000000001)
                self.minmax_etype[etype] = (minimum, maximum, R)
                
                if not src_type in costs_dict:
                    costs_dict[src_type] = (costs_array / R)
                    index_dict[src_type] = index_array
                else:
                    costs_dict[src_type] += (costs_array / R)
                    assert all(index_dict[src_type][0,:] == index_array[0,:])
                    assert all(index_dict[src_type][1,:] == index_array[1,:])
        print("_add_costs", time.time() - start_time)
        return costs_dict, index_dict
                    
    def _costs_of_merges(self, merge_list):
        start_time = time.time()
        
        
        self.init_costs_dict_features = self._feature_costs( merge_list)
        self.init_costs_dict_etype = self._h_costs( merge_list)    
        self.init_costs_dict, self.init_index_dict = self._add_costs(self.init_costs_dict_features, self.init_costs_dict_etype)
        print("_costs_of_merges", time.time() - start_time)
        return self.init_costs_dict , self.init_index_dict
            
            
            
        
       
    
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
        self.init_neighbors = self._get_union(init_costs)
        
        merge_costs, edges = self._costs_of_merges(self.init_neighbors)
        self._init_merge_graph(merge_costs, edges)
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
        


class Tester(): 
    def __init__(self):
        self.s = dict()
        self.s[0] = (2 / np.sqrt(2))
        self.s[1] = (1 / np.sqrt(2)) * (3 + 5) + (1 / np.sqrt(3)) * (4)
        self.s[2] = (1 / np.sqrt(3)) * (4) 
        self.s[3] = 0
        self.s[4] = 0
        
        self.h = dict()
        self.h[0] = 1.0
        self.h[1] = (1 / np.sqrt(4)) * self.s[1]
        self.h[2] = (1 / np.sqrt(2)) * self.s[2]
        self.h[3] = 0
        self.h[4] = 0
    
        self.nearest_neighbors = {0: {1,2}, 1:{0,2}, 2:{0,1}, 3:{2,4}, 4:{3}}
    def create_test_graph(self):
        g = dgl.heterograph({
            ('user', 'follows', 'user'): ([0, 1, 1, 1, 2], [1, 2, 3, 4,3])})
        g.nodes['user'].data['feat'] = torch.tensor([[1.0],[2.0],[3.0],[4.0],[5.0]])
        

        return g



    def check_H(self):
        for k, v in self.s.items():
            assert math.isclose(self.coarsener.coarsened_graph.nodes["user"].data[f'sfollows'][k].item(), v, rel_tol=1e-6), f"error in creating H for {k}"
            
        for k, v in self.h.items():
            assert math.isclose(self.coarsener.coarsened_graph.nodes["user"].data[f'hfollows'][k].item(), v, rel_tol=1e-6), f"error in creating H for {k}"
        
    def check_init_H_costs(self):
        neighbors = self.coarsener.init_neighbors["user"]
        correct = self.nearest_neighbors
        assert neighbors == correct, f"error in init neighbors {neighbors} != {correct}"
    
    def check_init_feat_costs(self):
        
        costs = self.coarsener.init_costs_dict_features["user"]["costs"]
        index = self.coarsener.init_costs_dict_features["user"]["index"]
        self.correct_feat_costs = {(0,1): 1, (1,0): 1 ,(2,0):2,   (0,2): 2, (1,2): 1, (2,1):1, (3,2): 1, (4,3) : 1, (3,4): 1}
        for i in range(costs.shape[0]):
            node1 = index[0][i].item()
            node2 = index[1][i].item()
            cost = costs[i].item()
            if (node1, node2) in self.correct_feat_costs:
                assert math.isclose(cost, self.correct_feat_costs[(node1, node2)], rel_tol=1e-6), f"error in init feat costs {cost} != {self.correct_feat_costs[(node1, node2)]}"
            else:
                assert math.isclose(cost, 0.0, rel_tol=1e-6), f"error in init feat costs {cost} != 0"
            assert len(self.correct_feat_costs) == costs.shape[0], f"error in init feat costs {costs.shape[0]} != {len(self.correct_feat_costs)}"
            
    def check_init_H_costs(self):
        
        self.correct_H_costs = { }
        for k, list_values in self.nearest_neighbors.items():
            for v in list_values:
                
                huv = (self.s[k] + self.s[v]) / np.sqrt(self.g.out_degrees(k) + self.g.out_degrees(v) + 2)
                self.correct_H_costs[k,v] =  torch.norm(torch.tensor(huv - self.h[k]),  p=1).item() + torch.norm(torch.tensor(huv -self.h[v]),  p=1).item()
        costs = self.coarsener.init_costs_dict_etype["user"]["follows"]["costs"]
        index = self.coarsener.init_costs_dict_etype["user"]["follows"]["index"]
        for i in range(costs.shape[0]):
            node1 = index[0][i].item()
            node2 = index[1][i].item()
            cost = costs[i].item()
            if (node1, node2) in self.correct_H_costs:
                assert math.isclose(cost, self.correct_H_costs[(node1, node2)], rel_tol=1e-2), f"error in init feat costs {cost} != {self.correct_H_costs[(node1, node2)]}"
            else:
                assert math.isclose(cost, 0.0, rel_tol=1e-6), f"error in init feat costs {cost} != 0"
        assert len(self.correct_H_costs) == costs.shape[0], f"error in init feat costs {costs.shape[0]} != {len(self.correct_H_costs)}"

    def check_init_total_costs(self):
        costs = self.coarsener.init_costs_dict["user"]
        index = self.coarsener.init_index_dict["user"]
        correct_costs = dict()
        for k, list_values in self.nearest_neighbors.items():
            for v in list_values:
                correct_costs[k,v] = (self.correct_H_costs[k,v] / 2.9831)  + (self.correct_feat_costs[k,v] / 1)
                
        for i in range(costs.shape[0]):
            node1 = index[0][i].item()
            node2 = index[1][i].item()
            cost = costs[i].item()
            if (node1, node2) in self.correct_H_costs:
                assert math.isclose(cost, correct_costs[(node1, node2)], rel_tol=1e-2), f"error in init feat costs {cost} != {correct_costs[(node1, node2)]}"
            else:
                assert math.isclose(cost, 0.0, rel_tol=1e-6), f"error in init feat costs {cost} != 0"
        assert len(self.correct_H_costs) == costs.shape[0], f"error in init feat costs {costs.shape[0]} != {len(self.correct_H_costs)}"

    def check_first_merge_candidates(self):
        candidates = [(4,3), (1,2)]
        assert self.coarsener.candidates["user"] == candidates, "error first merge candidates"
        
    def check_first_merge_nodes(self):
        assert all(self.coarsener.coarsened_graph.edges()[0] ==  torch.tensor([0,2])), "error merged edges not correct"
        assert all(self.coarsener.coarsened_graph.edges()[1] ==  torch.tensor([2,1]) ), "error merged edges not correct"
        
        assert all(self.coarsener.coarsened_graph.nodes["user"].data["feat"] == torch.tensor([[1.0], [4.5], [2.5]])), "error merge features not correct"
        
    def check_first_merge_s_and_h(self):
        correct_s ={
            0: self.s[0],
            2: (self.s[1] + self.s[2]),
            1: (self.s[3] + self.s[4]),
        }
        
        for k, v in correct_s.items():
            assert math.isclose(self.coarsener.coarsened_graph.nodes["user"].data[f'sfollows'][k].item(), v, rel_tol=1e-6), f"error in updating S for {k}"       
        
        correct_h = {
            0: correct_s[0] / np.sqrt(2),
            1: correct_s[1] / np.sqrt(2 ),
            2: correct_s[2] / np.sqrt(2 + 2),
        }
        for k, v in correct_h.items():
            assert math.isclose(self.coarsener.coarsened_graph.nodes["user"].data[f'hfollows'][k].item(), v, rel_tol=1e-6), f"error in updating H for {k}"
        
        pass
    
    def check_first_merge_features(self):
        correct_feat = {
            0: torch.tensor([1.0]),
            1: torch.tensor([4.5]),
            2: torch.tensor([2.5]),
        }
        for k, v in correct_feat.items():
            assert math.isclose(self.coarsener.coarsened_graph.nodes["user"].data["feat"][k].item(), v.item(), rel_tol=1e-6), f"error in updating features for {k}"
    
    def check_first_merge_updated_merge_edges(self):
        recalc_edges = (torch.tensor([0,1,2]), torch.tensor([2,2,0]))
        assert all(self.coarsener.merge_graphs["user"].find_edges(self.coarsener.edge_ids_need_recalc)[0] == recalc_edges[0]), "error in updated merge edges"
        assert all(self.coarsener.merge_graphs["user"].find_edges(self.coarsener.edge_ids_need_recalc)[1] == recalc_edges[1]), "error in updated merge edges"
        
        new_feature_costs = torch.tensor([1.5, 2, 1.5])
        assert  all(self.coarsener.costs_features == new_feature_costs), "error in updated merge edges"
        
        new_h_costs = torch.tensor([
            abs((self.s[0] + self.s[1] + self.s[2] )/ np.sqrt(5) - (self.s[0] / np.sqrt(2))) +  abs((self.s[0] + self.s[1] + self.s[2]) / np.sqrt(5) - ((self.s[1] + self.s[2]) / np.sqrt(4))),
            abs((self.s[1] + self.s[2] + self.s[3] + self.s[4] )/ np.sqrt(6) -  (self.s[1] + self.s[2]) / np.sqrt(4)) +  abs((self.s[1] + self.s[2] + self.s[3] + self.s[4]) / np.sqrt(6) - (self.s[3] + self.s[4] / np.sqrt(4))),
            abs((self.s[0] + self.s[1] + self.s[2] )/ np.sqrt(5) - self.s[0] / np.sqrt(2)) +  abs((self.s[0] + self.s[1] + self.s[2]) / np.sqrt(5) - ((self.s[1] + self.s[2]) / np.sqrt(4))),
        ])
        new_h_costs = new_h_costs / 2.9831
        for i in range(new_h_costs.shape[0]):
            assert math.isclose(new_h_costs[i].item(), self.coarsener.costs_H[i].item(), rel_tol=1e-2), f"error in updated merge edges {new_h_costs[i]} != {self.coarsener.costs_H[i]}"
        t = 2
        
    def run_test(self):
        self.g = self.create_test_graph()
        self.coarsener = HeteroCoarsener(None, self.g, 0.5, num_nearest_per_etype=2, num_nearest_neighbors=2,pairs_per_level=30)
        self.coarsener.init_step()
        self.check_H()
        self.check_init_H_costs()
        self.check_init_feat_costs()
        self.check_init_total_costs()
        self.check_first_merge_candidates()
        self.coarsener.iteration_step()
        self.check_first_merge_nodes()
        self.check_first_merge_s_and_h()
        self.check_first_merge_features()
        self.check_first_merge_updated_merge_edges()
        
        
    
    


if __name__ == "__main__":
    tester = Tester()
    tester.run_test()
    dataset = Citeseer() 
    original_graph = dataset.load_graph()

    #original_graph = create_test_graph()
    coarsener = HeteroCoarsener(None,original_graph, 0.5, num_nearest_per_etype=3, num_nearest_neighbors=3,pairs_per_level=30)
    coarsener.init_step()
    for i in range(600):
        print("--------- step: " , i , "---------" )
        coarsener.iteration_step()