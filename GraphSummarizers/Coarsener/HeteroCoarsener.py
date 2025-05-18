import copy
import dgl
import numpy as np
import os
import sys
import math
import time
import torch
from torch_scatter import scatter_mean
from sklearn.decomposition import PCA
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "../../"))
from Datasets.Dataset import Dataset
from test_hetero_coarsening import TestHomo, TestHetero
from sklearn.preprocessing import MinMaxScaler
from GraphSummarizers.GraphSummarizer import GraphSummarizer
from GraphSummarizers.GraphSummarizer import SUMMARY_GRAPH_FILENAME
from GraphSummarizers.GraphSummarizer import SUMMARY_STATISTICS_FILENAME
from GraphSummarizers.GraphSummarizer import ORIGINAL_TO_SUPER_NODE_FILENAME
from Datasets.NodeClassification.DBLP import DBLP
from Datasets.NodeClassification.AIFB import AIFB
from Datasets.NodeClassification.Citeseer import Citeseer
from torch_scatter import scatter_add
import pickle
import dgl
import torch
import scipy
from collections import Counter 
from copy import deepcopy
CHECKPOINTS = [0.7, 0.5, 0.3, 0.1, 0.05, 0.01, 0.001]

class HeteroCoarsener(GraphSummarizer):
    def __init__(self, dataset: Dataset, original_graph: dgl.DGLGraph, r: float, pairs_per_level: int = 10, 
                 num_nearest_neighbors: int = 10, num_nearest_per_etype:int = 10, filename = "dblp", R=None, device=None
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
        if device:
            self.device = device
        else: 
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.filename = filename
        assert (r > 0.0) and (r <= 1.0)
        self.r = r
        self.pca_components = 3
        self.original_graph = original_graph.to(self.device)
        self.dataset = dataset
        self.num_nearest_neighbors = num_nearest_neighbors
        self.num_nearest_per_etype = num_nearest_per_etype
        self.pairs_per_level = pairs_per_level
        self.coarsened_graph = original_graph.clone()
        self.coarsened_graph = self.coarsened_graph.to(self.device)
        self.merge_graph = None
        self.node_types = dict()
        self.candidate_pairs = None
        self.homo_hetero_map = dict()
        self.type_homo_mapping = dict()
        
        self.label_list_supernodes = dict()
        self.init_node_info()
        self.R = R
        self.minmax_ntype = dict()
        self.minmax_etype = dict()
        total_num_nodes = original_graph.num_nodes()
        self.ntype_distribution = dict()
        for ntype in original_graph.ntypes:
            self.ntype_distribution[ntype] = original_graph.num_nodes(ntype) / total_num_nodes
        
    
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
        """
        Vectorized GPU implementation of spatial RGCN coarsening.
        Returns H and S dicts with per-etype tensors.
        """
        
        start_time = time.time()

        for src_type, etype, dst_type in g.canonical_etypes:
            # Determine if we have node features
            has_feat = 'feat' in g.nodes[dst_type].data

            # Precompute normalized degrees
            deg_out = torch.tensor(self.node_degrees[etype]['out'], device=self.device) + 1.0
            deg_in  = torch.tensor(self.node_degrees[etype]['in'], device=self.device)  + 1.0
            inv_sqrt_out = torch.rsqrt(deg_out)
            inv_sqrt_in  = torch.rsqrt(deg_in)

            # Load features or use scalar 1
            if has_feat:
                feats = g.nodes[dst_type].data['feat'].to(self.device)
                feat_dim = feats.shape[1]
            else:
                # treat feature as scalar 1
                feat_dim = 1

            # Extract all edges of this type
            u, v = g.edges(etype=(src_type, etype, dst_type))
            u = u.to(self.device)
            v = v.to(self.device)

            # Gather destination feats & normalize
            if has_feat:
                feat_v = feats[v]                              # [E, D]
            else:
                feat_v = torch.ones((v.shape[0], 1), device=self.device)
            s_e = feat_v * inv_sqrt_in[v].unsqueeze(-1)       # [E, D]

            # Scatter-add to compute S at source nodes
            n_src = g.num_nodes(src_type)
            S_tensor = torch.zeros((n_src, feat_dim), device=self.device)
            S_tensor = S_tensor.index_add(0, u, s_e)

            # Compute H = D_out^{-1/2} * S
            H_tensor = inv_sqrt_out.unsqueeze(-1) * S_tensor

            # Store in coarsened_graph
            self.coarsened_graph.nodes[src_type].data[f's{etype}'] = S_tensor
            self.coarsened_graph.nodes[src_type].data[f'h{etype}'] = H_tensor
            self.coarsened_graph.nodes[src_type].data['node_size']  = torch.ones((n_src, 1), device=self.device)

        print("_create_h_spatial_rgcn", time.time() - start_time)
        
   

    
    def _init_merge_graph(self, costs, edges):
       
        start_time = time.time()
        self.merge_graphs =dict()
        for ntype in self.coarsened_graph.ntypes:
            self.merge_graphs [ntype] = dgl.graph(([], []), num_nodes=self.coarsened_graph.number_of_nodes(ntype=ntype), device=self.device)
            
            self.merge_graphs[ntype].add_edges(edges[ntype][0,:],  edges[ntype][1,:])
            self.merge_graphs[ntype].edata["edge_weight"] = costs[ntype]
            
        print("_init_merge_graph", time.time() - start_time)
        
   

   
    def _compute_embeddings(self, node_type, etype=None, use_feat=True):
        # Select the proper feature tensor
        if use_feat and 'feat' in self.coarsened_graph.nodes[node_type].data:
            H = self.coarsened_graph.nodes[node_type].data['feat'].float()
        elif etype is not None and f'h{etype}' in self.coarsened_graph.nodes[node_type].data:
            H = self.coarsened_graph.nodes[node_type].data[f'h{etype}'].float()
        else:
            # TODO this case should not happen
            pass

        H = H.to(self.device)

        # PCA via torch.pca_lowrank if needed
        # if H.size(1) > self.pca_components:
        #     # center and normalize
        #     mean = H.mean(dim=0, keepdim=True)
        #     std = H.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-4)
        #     Hn = (H - mean) / std
        #     # low rank PCA
        #     U, S, V = torch.pca_lowrank(Hn, q=self.pca_components)
        #     H = Hn @ V[:, :self.pca_components]

        return H

    def _query(self, H):
        # compute pairwise L1 distances
        dist_matrix = torch.cdist(H, H, p=1)
        # get k smallest distances (including self)
        k = min(self.num_nearest_per_etype, H.size(0))
        dists, idxs = torch.topk(dist_matrix, k, largest=False)
        return dists, idxs

    def _init_calculate_clostest_edges(self, src_type, etype, dst_type):
        H = self._compute_embeddings(src_type, etype=etype, use_feat=False)
        return self._query(H)

    def _init_calculate_clostest_features(self, ntype):
        H = self._compute_embeddings(ntype, use_feat=True)
        return self._query(H)

    def _get_union(self, init_costs):
        start_time = time.time()
        closest = {src: {} for src,_,_ in self.coarsened_graph.canonical_etypes}
        # edges
        for src, etype, _ in self.coarsened_graph.canonical_etypes:
            dists, idxs = init_costs[etype]
            for i, neighbors in enumerate(idxs):
                neigh = set(neighbors.tolist()) - {i}
                closest[src].setdefault(i, set()).update(neigh)
        # features
        for ntype in self.coarsened_graph.ntypes:
            dists, idxs = init_costs[ntype]
            for i, neighbors in enumerate(idxs):
                neigh = set(neighbors.tolist()) - {i}
                closest.setdefault(ntype, {})
                closest[ntype].setdefault(i, set()).update(neigh)
        print("_get_union", time.time() -start_time)
        return closest        
     
    def _init_costs(self):
        start_time = time.time()
        init_costs = dict()
        init_costs_t = dict()
        
        
        for src_type, etype, dst_type in self.coarsened_graph.canonical_etypes:
           init_costs[etype] = self._init_calculate_clostest_edges(src_type, etype, dst_type)
          
          
        for ntype in self.coarsened_graph.ntypes:
            if "feat" in self.coarsened_graph.nodes[dst_type].data:
                
                init_costs[ntype] = self._init_calculate_clostest_features(ntype)   
              
                  
              
        
        print("_init_costs", time.time() - start_time)
        return init_costs

    def _find_lowest_cost_edges(self):
        start_time = time.time()
        topk_non_overlapping_per_type = dict()
        for ntype in self.coarsened_graph.ntypes:
            if ntype not in self.merge_graphs:
                continue
            costs = self.merge_graphs[ntype].edata["edge_weight"]
            edges = self.merge_graphs[ntype].edges()
            k = min(self.num_nearest_neighbors * self.pairs_per_level, costs.shape[0])
            lowest_costs = torch.topk(costs, k, largest=False, sorted=True)
            
            # Vectorizing the loop
            topk_non_overlapping = []
            nodes = set()
            
            edge_indices = lowest_costs.indices
            src_nodes = edges[0][edge_indices].cpu().numpy()
            dst_nodes = edges[1][edge_indices].cpu().numpy()

            # Avoiding the need to loop over every edge individually
            for i in range(len(src_nodes)):
                if len(nodes) > (self.pairs_per_level * self.ntype_distribution[ntype]):
                    break
                src_node = src_nodes[i]
                dst_node = dst_nodes[i]
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
            g_new = deepcopy(g)
            nodes_u = torch.tensor([i for i, _ in node_pairs], dtype=torch.int64, device=self.device)
            nodes_v = torch.tensor([i for  _,i in node_pairs], dtype=torch.int64, device=self.device)
            mapping = torch.arange(0, g.num_nodes(ntype=node_type), device=self.device )
            num_pairs = len(node_pairs)
            num_nodes_before = g_new.num_nodes(ntype= node_type)
            old_feats = g_new.nodes[node_type].data["feat"]
            cu = g_new.nodes[node_type].data["node_size"][nodes_u]
            cv = g_new.nodes[node_type].data["node_size"][nodes_v]
            feat_u = old_feats[nodes_u]
            feat_v = old_feats[nodes_v]
            g_new.add_nodes(num_pairs, ntype=node_type)
            
            g_new.nodes[node_type].data["node_size"][num_nodes_before:] = cu + cv
            g_new.nodes[node_type].data["feat"][num_nodes_before:] = (feat_u * cu  + feat_v * cv ) / (cu + cv)
            new_nodes = g_new.nodes(ntype= node_type)[num_nodes_before:]
            
            
            mapping[nodes_u] = g_new.nodes(ntype=node_type)[num_nodes_before:]
            mapping[nodes_v] = g_new.nodes(ntype=node_type)[num_nodes_before:]
            counts_u = (mapping.unsqueeze(1) > nodes_u).sum(dim=1) 
            counts_v = (mapping.unsqueeze(1) > nodes_v).sum(dim=1) 
            
            mapping = mapping - counts_u - counts_v
            for src_type, etype,dst_type in g.canonical_etypes:
                if src_type != node_type:
                    continue
                suv = g.nodes[node_type].data[f's{etype}'][nodes_u] + g.nodes[node_type].data[f's{etype}'][nodes_v]
                cuv = g.nodes[node_type].data["node_size"][nodes_u] + g.nodes[node_type].data["node_size"][nodes_v]
                g_new.nodes[node_type].data["node_size"][new_nodes] = cuv
                edges_original = g_new.edges(etype=etype)
                repeat_u = nodes_u.unsqueeze(1).repeat(1, edges_original[0].shape[0])
                repeat_v = nodes_v.unsqueeze(1).repeat(1, edges_original[0].shape[0])
                edges_u = repeat_u == edges_original[0].unsqueeze(0).repeat(repeat_u.shape[0], 1) 
                edges_v = repeat_v == edges_original[1].unsqueeze(0).repeat(repeat_v.shape[0], 1)
                edges_uv = torch.logical_or(edges_u, edges_v)
                duv = edges_uv.sum(dim=1)
                #duv = torch.logical_or(edges_u, edges_v, dim=0).sum()
                #duv = g.out_degrees(nodes_u, etype=etype) + g.out_degrees(nodes_v, etype=etype) # TODO
                g_new.nodes[node_type].data[f's{etype}'][new_nodes] = suv  
                g_new.nodes[node_type].data[f'h{etype}'][new_nodes] =  suv / torch.sqrt(duv.unsqueeze(1) + cuv)
            nodes_to_delete = torch.cat([nodes_u, nodes_v])           
            g_new.remove_nodes(nodes_to_delete, ntype=node_type)
                
            print("_merge_nodes", time.time() - start_time)
            return g_new, mapping
        
    def _add_merged_edges(self, g_before, g_after, mappings):
        for src_type, etype,dst_type in g_before.canonical_etypes:
            mapping_src = mappings[src_type][-1]
            mapping_dst = mappings[dst_type][-1]
            all_eids = g_after.edges(form='eid', etype=etype)
            g_after.remove_edges(all_eids, etype=etype)
            edges_original = g_before.edges(etype=etype)
            new_edges = torch.stack((mapping_src[edges_original[0]], mapping_dst[edges_original[1]]))
            new_edges = torch.unique(new_edges, dim=1)
            g_after.add_edges(new_edges[0], new_edges[1], etype=(src_type, etype, dst_type))
            if src_type == dst_type:
                g_after = dgl.remove_self_loop(g_after, etype=etype)
        return g_after
            
                
    def _update_merge_graph_nodes_edges(self, g, node_pairs):
        start_time = time.time()
        g_new = deepcopy(g)
        nodes_u = torch.tensor([i for i, _ in node_pairs], dtype=torch.int64, device=self.device)
        nodes_v = torch.tensor([i for  _,i in node_pairs], dtype=torch.int64, device=self.device)
        mapping = torch.arange(0, g.num_nodes() , device=self.device)
        num_pairs = len(node_pairs)
        num_nodes_before = g_new.num_nodes()
        g_new.add_nodes(num_pairs)
        
        
        mapping[nodes_u] = g_new.nodes()[num_nodes_before:]
        mapping[nodes_v] = g_new.nodes()[num_nodes_before:]
        counts_u = (mapping.unsqueeze(1) > nodes_u).sum(dim=1) 
        counts_v = (mapping.unsqueeze(1) > nodes_v).sum(dim=1) 
        
        mapping = mapping - counts_u - counts_v
        
        nodes_to_delete = torch.cat([nodes_u, nodes_v])
        g_new.remove_nodes(nodes_to_delete)
            
        all_eids = g_new.edges(form='eid')
        g_new.remove_edges(all_eids)
        edges_original = g.edges()
        new_edges = torch.stack((mapping[edges_original[0]], mapping[edges_original[1]]))
        new_edges = torch.unique(new_edges, dim=1)
        
        g_new.add_edges(new_edges[0], new_edges[1])
        g_new.edata["needs_check"] = torch.zeros(g_new.num_edges(), dtype=torch.bool, device=self.device)
        g_new.edata["needs_check"][g_new.edge_ids(new_edges[0], new_edges[1])] = True
        g_new = dgl.remove_self_loop(g_new)
        print("_update_merge_graph_nodes_vec", time.time() - start_time)
        return g_new
            
   

    def _update_merge_graph_edge_weigths_features(self, g, ntype, eids ):
        start_time = time.time()
        if "feat" not in self.coarsened_graph.nodes[ntype].data:
            return
        feat = self.coarsened_graph.nodes[ntype].data['feat']
        
        # 1) Build flat src/dst lists
        src , dst = g.find_edges(eids)  
      
        # 2) mask out self‐pairs (if any)
        mask = src != dst
        src, dst = src[mask], dst[mask]                                         # [E']

        # 3) pull features and compute costs
        f1 = feat[src]       # [E'×H]
        f2 = feat[dst]       # [E'×H]
        
        cu = self.coarsened_graph.nodes[ntype].data["node_size"][src]
        cv = self.coarsened_graph.nodes[ntype].data["node_size"][dst]        
        
        mid = (f1* cu + f2* cv) / (cu + cv)   # [E'×H]
        if self.R:
            costs = (torch.norm(mid - f1,  dim=1, p =1) + torch.norm(mid - f2,  dim=1, p=1))  * (self.R[ntype])
        else:
            costs = (torch.norm(mid - f1,  dim=1, p =1) + torch.norm(mid - f2,  dim=1, p=1))  / (self.minmax_ntype[ntype][2])  # [E']
        print("_update_merge_graph_edge_weigths_features", time.time()- start_time)
        return  costs
        
        
   

    def _update_merge_graph_edge_weights_H(self,g, ntype, eids):
        
        start_time = time.time()
        
        
        src_nodes , dst_nodes = g.find_edges(eids)  
        costs = torch.zeros(len(src_nodes), device=self.device)
        
        for src_type, etype, dst_type in self.coarsened_graph.canonical_etypes:
            if src_type != ntype:
                continue
            
            merged_repr = self._create_h_via_cache_vec_fast_without_table(src_nodes, dst_nodes, src_type, etype, self.coarsened_graph.nodes[src_type].data['node_size'])
            
            #merged_repr = torch.cat([H_merged[n] for n in candidates.keys()], dim=0)  

            src_repr = self.coarsened_graph.nodes[src_type].data[f'h{etype}'][src_nodes]   # TODO: wrong !!!
            dst_repr = self.coarsened_graph.nodes[src_type].data[f'h{etype}'][dst_nodes]   

            cost_src =torch.norm(src_repr - merged_repr,  dim=1, p=1)  # [E]
            cost_dst = torch.norm(dst_repr - merged_repr,  dim=1, p=1)  # [E]
            if self.R:
                total_cost = (cost_src + cost_dst) * (self.R[etype])
            else:
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
        
        g_before = deepcopy(go)
        g_coar = self._update_merge_graph_nodes_edges(g_before, node_pairs)
        self.edge_ids_need_recalc = g_coar.edata["needs_check"].nonzero(as_tuple=True)[0]
        
        if len(self.edge_ids_need_recalc) > 0:    
            self.costs_features = self._update_merge_graph_edge_weigths_features(g_coar,ntype, self.edge_ids_need_recalc)
            
            self.costs_H = self._update_merge_graph_edge_weights_H(g_coar, ntype, self.edge_ids_need_recalc)
            
            g_coar.edata["edge_weight"][self.edge_ids_need_recalc] = self.costs_features 
            g_coar.edata["edge_weight"][self.edge_ids_need_recalc] += self.costs_H
            print("_update_merge_graph", time.time()- start_time)    
            return g_coar,   True
        else:
            print("_update_merge_graph: WARNING no more merge candidates", time.time()- start_time)    
            return g_coar,  False
           
      
    def _create_h_via_cache_vec_fast(self,  table,ntype, etype, cluster_sizes):
        cache = self.coarsened_graph.nodes[ntype].data[f's{etype}']
        H_merged = dict()
        
        # 1) Flatten your table into two 1-D lists of equal length L:
        pairs = [(u, v) for u, vs in table.items() for v in vs]
        node1s, node2s = zip(*pairs)
        node1s = torch.tensor(node1s, dtype=torch.long)
        node2s = torch.tensor(node2s, dtype=torch.long)

        # 2) Grab degrees and caches in one go:
        deg = self.coarsened_graph.out_degrees(etype=etype)
        deg1 = deg[node1s]            # shape (L,)
        deg2 = deg[node2s]            # shape (L,)

        su = cache[node1s]            # shape (L, D)
        sv = cache[node2s]            # shape (L, D)

        # 3) Cluster‐size term (make sure cluster_sizes is a tensor):
        #csize = torch.tensor([cluster_sizes[i] for i in range(self.coarsened_graph.num_nodes())],
        #                    device=deg.device, dtype=deg.dtype)
        cuv = cluster_sizes[node1s] + cluster_sizes[node2s]  # shape (L,)

        # 4) Single vectorized compute of h for all L pairs:
        #    (we broadcast / unsqueeze cuv into the right D-dimensional form)
        h_all = (su + sv) / torch.sqrt((deg1 + deg2 + cuv.squeeze())).unsqueeze(1) #)  # (L, D)

        return h_all
   
    def _create_h_via_cache_vec_fast_without_table(self,  node1s, node2s,ntype, etype, cluster_sizes):
        cache = self.coarsened_graph.nodes[ntype].data[f's{etype}']
        H_merged = dict()
        
        # 1) Flatten your table into two 1-D lists of equal length L:
        node1s = torch.tensor(node1s, dtype=torch.long)
        node2s = torch.tensor(node2s, dtype=torch.long)

        # 2) Grab degrees and caches in one go:
        deg = self.coarsened_graph.out_degrees(etype=etype)
        deg1 = deg[node1s]            # shape (L,)
        deg2 = deg[node2s]            # shape (L,)

        su = cache[node1s]            # shape (L, D)
        sv = cache[node2s]            # shape (L, D)

        # 3) Cluster‐size term (make sure cluster_sizes is a tensor):
        #csize = torch.tensor([cluster_sizes[i] for i in range(self.coarsened_graph.num_nodes())],
        #                    device=deg.device, dtype=deg.dtype)
        cuv = cluster_sizes[node1s] + cluster_sizes[node2s]  # shape (L,)

        # 4) Single vectorized compute of h for all L pairs:
        #    (we broadcast / unsqueeze cuv into the right D-dimensional form)
        h_all = (su + sv) / torch.sqrt((deg1 + deg2 + cuv.squeeze())).unsqueeze(1) #)  # (L, D)

        return h_all
    
    def _create_h_via_cache_vec(self,  table,ntype, etype, cluster_sizes):
        cache = self.coarsened_graph.nodes[ntype].data[f's{etype}']
        H_merged = dict()
        
        # 1) Flatten your table into two 1-D lists of equal length L:
        pairs = [(u, v) for u, vs in table.items() for v in vs]
        node1s, node2s = zip(*pairs)
        node1s = torch.tensor(node1s, dtype=torch.long)
        node2s = torch.tensor(node2s, dtype=torch.long)

        # 2) Grab degrees and caches in one go:
        deg = self.coarsened_graph.out_degrees(etype=etype)
        deg1 = deg[node1s]            # shape (L,)
        deg2 = deg[node2s]            # shape (L,)

        su = cache[node1s]            # shape (L, D)
        sv = cache[node2s]            # shape (L, D)

        # 3) Cluster‐size term (make sure cluster_sizes is a tensor):
        #csize = torch.tensor([cluster_sizes[i] for i in range(self.coarsened_graph.num_nodes())],
        #                    device=deg.device, dtype=deg.dtype)
        cuv = cluster_sizes[node1s] + cluster_sizes[node2s]  # shape (L,)

        # 4) Single vectorized compute of h for all L pairs:
        #    (we broadcast / unsqueeze cuv into the right D-dimensional form)
        h_all = (su + sv) / torch.sqrt((deg1 + deg2 + cuv.squeeze())).unsqueeze(1) #)  # (L, D)

        
        #    and reassign:
        for pair, h_chunk in zip(pairs, h_all):
            node1 = pair[0]
            if node1 not in H_merged:
                H_merged[node1] = h_chunk.unsqueeze(0)
            else:
                H_merged[node1] = torch.cat((H_merged[node1], h_chunk.unsqueeze(0)), dim=0)
        return H_merged
    
    def _feature_costs(self, merge_list):
        costs_dict = {}
        for ntype in self.coarsened_graph.ntypes:
            data = self.coarsened_graph.nodes[ntype].data
            if "feat" not in data:
                continue

            feat = data["feat"]                   # [N, F]
            size = data["node_size"].to(self.device)   # [N]

            # build flat list of all valid (u,v) pairs
            starts, ends = [], []
            for u, vs in merge_list[ntype].items():
                # filter out self‐merges if any
                vs = [v for v in vs if v != u]
                if not vs:
                    continue
                starts.append(torch.full((len(vs),), u, dtype=torch.long, device=self.device))
                ends.append(torch.tensor(vs, dtype=torch.long, device=self.device))

            if not starts:
                continue

            node1_ids = torch.cat(starts)  # [P]
            node2_ids = torch.cat(ends)    # [P]

            # gather features & sizes
            f1 = feat[node1_ids]            # [P, F]
            f2 = feat[node2_ids]            # [P, F]
            s1 = size[node1_ids]#.unsqueeze(1)  # [P, 1]
            s2 = size[node2_ids]#.unsqueeze(1)  # [P, 1]

            # merged feature
            merged = (f1 * s1 + f2 * s2) / (s1 + s2)  # [P, F]

            # L1 cost
            cost = torch.norm(merged - f1, p=1, dim=1) + torch.norm(merged - f2, p=1, dim=1)  # [P]

            costs_dict[ntype] = {
                "costs": cost,
                "index": torch.stack([node1_ids, node2_ids], dim=0)
            }
        return costs_dict

    
    def _h_costs(self, merge_list):
        costs_dict = {}
        for src, etype, dst in self.coarsened_graph.canonical_etypes:
            # ensure nested dict
            costs_dict.setdefault(src, {})[etype] = {}
            # compute all merged h representations in one go
            H_merged = self._create_h_via_cache_vec_fast(
                merge_list[src], src, etype,
                torch.ones(self.coarsened_graph.number_of_nodes(src), device=self.device)
            )  # [N_src, hidden]

            # flatten all (u,v) pairs same as above
            starts, ends = [], []
            for u, vs in merge_list[src].items():
                vs = [v for v in vs if v != u]
                if not vs:
                    continue
                starts.append(torch.full((len(vs),), u, dtype=torch.long, device=self.device))
                ends.append(torch.tensor(vs, dtype=torch.long, device=self.device))

            if not starts:
                continue

            node1_ids = torch.cat(starts)  # [P]
            node2_ids = torch.cat(ends)    # [P]

            # gather representations
            h1 = self.coarsened_graph.nodes[src].data[f"h{etype}"][node1_ids]  # [P, H]
            h2 = self.coarsened_graph.nodes[src].data[f"h{etype}"][node2_ids]  # [P, H]
             # build a dense [num_src, hidden] tensor
            #H_tensor =  torch.tensor([v for k,v in  H_merged.items()] , device=device)
            merged = H_merged                               # [P, H]

            # L1 costs
            cost = torch.norm(merged - h1, p=1, dim=1) + torch.norm(merged - h2, p=1, dim=1)
        #    cost = (h1 - merged).abs().sum(dim=1) + (h2 - merged).abs().sum(dim=1)  # [P]

            costs_dict[src][etype] = {
                "costs": cost,
                "index": torch.stack([node1_ids, node2_ids], dim=0)
            }
        return costs_dict

 

    def _add_costs(self, feat_costs, etype_costs):
        # normalize and accumulate
        costs_dict, index_dict = {}, {}

        # feature costs
        for ntype, cd in feat_costs.items():
            cost = cd["costs"]
            mn, mx = cost.min(), cost.max()
            R = mx - mn + 1e-13
            self.minmax_ntype[ntype] = (mn, mx, R)
            if self.R:
                
                norm = cost * (self.R.get(ntype, R))
            
            else:
                norm = cost / R
            costs_dict[ntype] = norm
            index_dict[ntype] = cd["index"]

        # etype costs
        for src, etypes in etype_costs.items():
            for etype, cd in etypes.items():
                cost = cd["costs"]
                mn, mx = cost.min(), cost.max()
                R = mx - mn + 1e-13
                self.minmax_etype[etype] = (mn, mx, R)
                if self.R:
                    norm = cost * self.R.get(etype, R)
                else: 
                    norm = cost / R
                if src in costs_dict:
                    # broadcast add onto existing vector
                    costs_dict[src] = costs_dict[src] + norm
                    # indices must match
                    assert torch.equal(index_dict[src], cd["index"])
                else:
                    costs_dict[src] = norm
                    index_dict[src] = cd["index"]

        return costs_dict, index_dict
                    
    def _costs_of_merges(self, merge_list):
        start_time = time.time()
        
      
        self.init_costs_dict_features = self._feature_costs( merge_list)
        
        self.init_costs_dict_etype = self._h_costs( merge_list)    
        
        self.init_costs_dict, self.init_index_dict = self._add_costs(self.init_costs_dict_features, self.init_costs_dict_etype)
       
        print("_costs_of_merges", time.time() - start_time)
        
        return self.init_costs_dict , self.init_index_dict
            
            
            
    def _get_master_mapping_tensor(self, mappings, ntype):
        master_mapping = dict()
        nodes_orig = self.original_graph.nodes(ntype)
        nodes = self.original_graph.nodes(ntype)
        for mapping in mappings:
            nodes = mapping[nodes]
        for i in range(len(nodes)):
            master_mapping[nodes_orig[i].item()] = nodes[i].item()
        
        return master_mapping    
        
        
        for node in self.original_graph.nodes(ntype):
            node_id = node.item()
            for mapping in mappings:
                node_id = mapping[node_id].item()
            master_mapping[node.item()] = node_id
        return master_mapping
        
        return master_mapping  # both are 1D tensors of the same length

       
    
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
        
        
       
                
        init_costs = self._init_costs()
        
        self.init_neighbors = self._get_union(init_costs)
        
        merge_costs, edges = self._costs_of_merges(self.init_neighbors)
        self._init_merge_graph(merge_costs, edges)
        
        self.candidates = self._find_lowest_cost_edges()
       
   
   
   #     pickle.dump(self, open(file_name, "wb"))
    

    
    
    def iteration_step(self):
        isNewMerges = False
        g1 = deepcopy(self.coarsened_graph)
        if self.original_graph.number_of_nodes() * self.r >  self.coarsened_graph.number_of_nodes():
            return False
        for ntype, merge_list in self.candidates.items():
            self.coarsened_graph, mapping = self._merge_nodes(self.coarsened_graph, ntype, merge_list)
            self.mappings[ntype].append(mapping) 
           # 
                
        
        self.coarsened_graph = self._add_merged_edges(g1,self.coarsened_graph, self.mappings)
        
        for ntype, merge_list in self.candidates.items():
            self.merge_graphs[ntype], isNewMergesPerType = self._update_merge_graph(self.merge_graphs[ntype], merge_list, ntype)
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
        t = self._get_master_mapping_tensor(self.mappings[ntype], ntype)
    #    t2 = self._get_master_mapping(self.mappings[ntype], ntype )
        
        return self._get_master_mapping_tensor(self.mappings[ntype], ntype )
        



if __name__ == "__main__":
    tester = TestHetero()
    g = tester.g 
    tester.run_test(HeteroCoarsener(None,g, 0.5, num_nearest_per_etype=2, num_nearest_neighbors=2,pairs_per_level=30, device="cpu"))
    dataset = DBLP() 
    original_graph = dataset.load_graph()

    #original_graph = create_test_graph()
    coarsener = HeteroCoarsener(None,original_graph, 0.5, num_nearest_per_etype=3, num_nearest_neighbors=3,pairs_per_level=30)
    coarsener.init_step()
    for i in range(3):
        print("--------- step: " , i , "---------" )
        coarsener.iteration_step()
        coarsener.get_mapping("author")