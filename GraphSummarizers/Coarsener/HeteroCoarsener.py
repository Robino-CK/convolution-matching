import copy
import dgl
import numpy as np
from tqdm import tqdm    
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
                 num_nearest_neighbors: int = 10, num_nearest_per_etype:int = 10, filename = "dblp", R=None, device=None, is_neighboring_h = False, is_eval_metrics=False,
                 is_adj=False):
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
            self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.filename = filename
        assert (r > 0.0) and (r <= 1.0)
        self.r = r
        self.is_eval_metrics = is_eval_metrics
        if self.is_eval_metrics:
            self.evaluations_distances = []
            self.feat_distances = dict()
            self.feat_scores = dict()
            self.edge_distances = dict()
            self.edge_scores = dict()
        self.is_adj = is_adj
        self.means = dict()
        self.stds = dict()
        
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
        self.is_neighboring_h = is_neighboring_h
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
            # for src_type, etype, dst_type in self.coarsened_graph.canonical_etypes:
            #     self.coarsened_graph.edata[f"out_adj_{etype}"] = torch.ones(self.coarsened_graph.add_edges(etype=etype), device=self.device)
                
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
            #deg_in  = torch.tensor(self.node_degrees[etype]['in'], device=self.device)  + 1.0
            inv_sqrt_out = torch.rsqrt(deg_out)
            #inv_sqrt_in  = torch.rsqrt(deg_in)
            

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
                
            
            s_e = feat_v * inv_sqrt_out[v].unsqueeze(-1)       # [E, D]
            # Scatter-add to compute S at source nodes
            n_src = g.num_nodes(src_type)
            S_tensor = torch.zeros((n_src, feat_dim), device=self.device)
            S_tensor = S_tensor.index_add(0, u, s_e)
            infl = torch.zeros(n_src, device=self.device)
            infl = infl.index_add(0, u, inv_sqrt_out[v])

            # Compute H = D_out^{-1/2} * S
            H_tensor = inv_sqrt_out.unsqueeze(-1) * S_tensor

            # Store in coarsened_graph
            self.coarsened_graph.nodes[src_type].data[f'i{etype}'] = infl
            
            self.coarsened_graph.nodes[src_type].data[f's{etype}'] = S_tensor
            #print( ((1 / (deg_out.unsqueeze(-1))) * feats).shape)
            print(H_tensor.shape)
            self.coarsened_graph.nodes[src_type].data[f'h{etype}'] = H_tensor + ((feats / (deg_out.unsqueeze(-1))))
            self.coarsened_graph.nodes[src_type].data['node_size']  = torch.ones((n_src, 1), device=self.device)

        print("_create_h_spatial_rgcn", time.time() - start_time)
        
   

    
    def _init_merge_graph(self, init_costs_dict_features, init_costs_dict_etype, init_costs_dict_etype_neighbors):
       
        start_time = time.time()
        self.merge_graphs =dict()
        for ntype in self.coarsened_graph.ntypes:
            
         #   d_total = torch.zeros(init_costs_dict_features[ntype]["index"].shape[1] ,device=self.device)
          #  for src_type, etype, dst_type in self.coarsened_graph.canonical_etypes:
           #     if src_type == ntype:
            #        d_total += self.coarsened_graph.out_degrees(init_costs_dict_features[ntype]["index"][0,:], etype=etype)
            
            
            self.merge_graphs [ntype] = dgl.graph(([], []), num_nodes=self.coarsened_graph.number_of_nodes(ntype=ntype), device=self.device)
            
            
            edges_add = False
            if ntype in init_costs_dict_features:
                costs = init_costs_dict_features[ntype]["costs"]
                edges = init_costs_dict_features[ntype]["index"]
                self.merge_graphs[ntype].add_edges(edges[0,:],  edges[1,:])
                self.merge_graphs[ntype].edata["edge_weight_feat"] = costs 
                edges_add = True
                score = self.zscore(costs, ntype )
                if self.R:
                    score = score * self.R[ntype]
                if self.is_eval_metrics:
                    self.feat_scores[ntype] = []
                    self.feat_scores[ntype].append(score)
                self.merge_graphs[ntype].edata["edge_weight"] = score
                
            else: 
                self.merge_graphs[ntype].edata["edge_weight"] = torch.zeros(self.merge_graphs[ntype].num_edges(), device=self.device)
                
            for etype, costs_edges in init_costs_dict_etype[ntype].items():
                if not edges_add:
                    edges = init_costs_dict_etype[ntype][etype]["index"]
                    self.merge_graphs[ntype].add_edges(edges[0,:],  edges[1,:])
                    edges_add = True
                
                
                    
              #  d_etype = self.coarsened_graph.out_degrees(init_costs_dict_features[ntype]["index"][0,:], etype=etype)
                costs = costs_edges["costs"] 
                self.merge_graphs[ntype].edata[f"edge_weight_{etype}"] = costs
                score = self.zscore(costs, etype )
                if self.R:
                    score = score * self.R[etype]
                if self.is_eval_metrics:
                    self.edge_scores[etype] = []
                    self.edge_scores[etype].append(score)
                self.merge_graphs[ntype].edata["edge_weight"] += score#* d_etype/ d_total
                
        for ntype in self.coarsened_graph.ntypes:
            if init_costs_dict_etype_neighbors:
                if ntype in init_costs_dict_etype_neighbors:
                    for etype, costs_edges in init_costs_dict_etype_neighbors[ntype].items():
                        costs = costs_edges["costs"]
                        self.merge_graphs[ntype].edata[f"edge_neig_weight_{etype}"] = costs
                        score = self.zscore(costs, etype , False)
                        if self.R:
                            score = score * self.R[etype]
                        self.merge_graphs[ntype].edata["edge_weight"] += score
                    
        
                    
            
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
            if not ntype in init_costs.keys():
                continue 
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
        #    if self.is_eval_metrics and False:
                # Step 1: Convert to homogeneous graph (merge all edges into one)
                
                # Compute distances for each pair
             #   
            mapping = torch.arange(0, g.num_nodes(ntype=node_type), device=self.device )
            num_pairs = len(node_pairs)
            num_nodes_before = g_new.num_nodes(ntype= node_type)
            if "feat" in  g_new.nodes[node_type].data:
                old_feats = g_new.nodes[node_type].data["feat"]
                feat_u = old_feats[nodes_u]
                feat_v = old_feats[nodes_v]
            cu = g_new.nodes[node_type].data["node_size"][nodes_u]
            cv = g_new.nodes[node_type].data["node_size"][nodes_v]
            
            g_new.add_nodes(num_pairs, ntype=node_type)
            
            g_new.nodes[node_type].data["node_size"][num_nodes_before:] = cu + cv
            if "feat" in  g_new.nodes[node_type].data:
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
                cu = g.nodes[node_type].data["node_size"][nodes_u]
                cv = g.nodes[node_type].data["node_size"][nodes_v]
                suv = g.nodes[node_type].data[f's{etype}'][nodes_u]  + g.nodes[node_type].data[f's{etype}'][nodes_v] 
                cuv = cu + cv
                g_new.nodes[node_type].data["node_size"][new_nodes] = cuv
                edges_original = g_new.edges(etype=etype)
                repeat_src = nodes_u.unsqueeze(1).repeat(1, edges_original[0].shape[0])
                repeat_dst = nodes_v.unsqueeze(1).repeat(1, edges_original[0].shape[0])
                edges_src = repeat_src == edges_original[0].unsqueeze(0).repeat(repeat_src.shape[0], 1) 
                edges_dst = repeat_dst == edges_original[0].unsqueeze(0).repeat(repeat_dst.shape[0], 1)
                
                edges_src_dst = torch.logical_or(edges_src, edges_dst)
                duv = edges_src_dst.sum(dim=1)
                #duv = edges_src.sum(dim=1) + edges_dst.sum(dim=1)
                
                infl_u =  g.nodes[node_type].data[f'i{etype}'][nodes_u]
                infl_v = g.nodes[node_type].data[f'i{etype}'][nodes_v]
                adj_u = self._get_adj(nodes_u, nodes_v,etype)
                adj_v = self._get_adj(nodes_v, nodes_u,etype)
                deg = self._get_degree_with_adj(nodes_u, nodes_v, etype)
                infl_uv =   infl_u - (adj_u / torch.sqrt(cv.squeeze() + deg[nodes_v])) + infl_v -  (adj_v / torch.sqrt(cu.squeeze() + deg[nodes_u])) 
                #infl_uv = (infl_u  * cu.squeeze() + infl_v * cv.squeeze()) / cuv.squeeze()# - (1 / torch.sqrt(edges_v.sum(dim=1) +  cv.squeeze()))
                exists = g.has_edges_between(nodes_u, nodes_v, etype=etype)
                adj = torch.zeros(len(nodes_u), device=self.device)
                                # Get indices where the edge exists
                existing_edge_indices = exists.nonzero(as_tuple=True)[0]


                # Get edge IDs for the existing edges
                edge_ids = g.edge_ids(nodes_u[existing_edge_indices], nodes_v[existing_edge_indices], etype=etype)
                g_new.nodes[node_type].data[f'i{etype}'][new_nodes] =   infl_uv
                
                neighbors, nodes, eids = self.coarsened_graph.in_edges( torch.cat((nodes_v, nodes_u)) ,form="all", etype=etype)
                
                unique_nodes, inverse_indices = torch.unique(nodes, return_inverse=True)
                
                pos_v = torch.searchsorted(unique_nodes, nodes_v)
                pos_u = torch.searchsorted(unique_nodes, nodes_u)

                # Create boolean masks for each original node set
                mask_v = (inverse_indices.unsqueeze(0) == pos_v.unsqueeze(1))
                mask_u = (inverse_indices.unsqueeze(0) == pos_u.unsqueeze(1))

                # Get indices where each node appears
                indices_v = torch.where(mask_v)  # (node_idx, position_in_nodes)
                indices_u = torch.where(mask_u)  # (node_idx, position_in_nodes)
          #      neighbors = torch.cat(neighbors_v, neighbors_u)
                #g_new.edges[etype].data['adj'][eid_v]
                pairs = torch.stack([neighbors, nodes], dim=1)
                unique_edges, inverse_edges = torch.unique(pairs, return_inverse=True, dim=0)
                
                #unique_uv, inverse_uv = torch.unique(torch.cat((neighbors_u, neighbors_v)), return_inverse=True)
                
                adj = self._get_adj(neighbors, nodes,etype)
                unique_neigbors, inverse_neigbors = torch.unique(neighbors, return_inverse=True)
                
                c = g.nodes[node_type].data["node_size"][nodes]
             
                mask_n_u = torch.isin(nodes, nodes_u)
                adj_u = adj[mask_n_u]
                deg_u = deg[nodes[mask_n_u]]
                c_u = c[mask_n_u]
                new_neighbors_infl = torch.zeros(len(neighbors), device = self.device)#torch.zeros(len(unique_neigbors), device=self.device )
                term_u = - (adj_u / torch.sqrt(deg_u + c_u.squeeze()))
                new_neighbors_infl[mask_n_u] += term_u
                
                mask_n_v = torch.isin(nodes, nodes_v)
                adj_v = adj[mask_n_v]
                deg_v = deg[nodes[mask_n_v]]
                c_v = c[mask_n_v]
                term_v = - (adj_v / torch.sqrt(deg_v + c_v.squeeze()))
                new_neighbors_infl[mask_n_v] += term_v
                
                uni_neighbors_influence = g_new.nodes[node_type].data[f'i{etype}'][unique_neigbors]
                uni_neighbors_influence = uni_neighbors_influence.index_add(0, inverse_neigbors, new_neighbors_infl)
                
                
                cuv = g.nodes[node_type].data["node_size"][nodes_u].squeeze() + g.nodes[node_type].data["node_size"][nodes_v].squeeze()
                deguv = deg[nodes_u] + deg[nodes_v]
                
                mask_u = torch.isin(nodes_u, nodes)
                mask_v = torch.isin(nodes_v, nodes)
                
                adj = g.edges[etype].data['adj'][eids]
                nodes
                
                
                
                term_uv =  (adj / torch.sqrt(deg[nodes_u] + deg[nodes_v] + c.squeeze()))
                uni_neighbors_influence = uni_neighbors_influence.index_add(0, inverse_neigbors, term_uv)
                edge_data = g.edges[etype].data['adj'][edge_ids]

                # Assign edge data to the result tensor
                adj[existing_edge_indices] = edge_data
            
                du = edges_src.sum(dim=1)
                dv = edges_dst.sum(dim=1)
                
                minus = torch.mul((adj / torch.sqrt(du + cu.squeeze() )).unsqueeze(1), feat_u) + torch.mul((adj / torch.sqrt(dv + cv.squeeze() )).unsqueeze(1), feat_v) # + torch.matmul( (adj / torch.sqrt(deg2 + cv.squeeze() )), feat_v)
        
                
                g_new.nodes[node_type].data[f's{etype}'][new_nodes] = suv  - minus
                
                
                g_new.nodes[node_type].data[f'h{etype}'][new_nodes] =  (suv  - minus)  / torch.sqrt(duv.unsqueeze(1) + cuv) + ((feat_u * cu  + feat_v * cv ) /   (duv.unsqueeze(1) + cuv)) 
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
            edges_adj = g_before.edges[etype].data["adj"]
            new_edges = torch.stack((mapping_src[edges_original[0]], mapping_dst[edges_original[1]]))
            pairs = torch.stack((new_edges[0], new_edges[1]), dim=1)

            uniq_pairs, inverse = torch.unique(
               pairs, dim=0, return_inverse=True
                )
            sums = torch.zeros(len(uniq_pairs), dtype=edges_adj.dtype, device=self.device)
            sums.index_add_(0, inverse, edges_adj) 
           # new_edges, counts = torch.unique(new_edges, return_counts=True, dim=1)
            #new_edges = torch.unique(new_edges, dim=1)
            
            eids = g_after.add_edges(uniq_pairs[:,0], uniq_pairs[:,1], etype=(src_type, etype, dst_type))
            g_after.edges[etype].data["adj"][eids] = sums
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
        #print("_update_merge_graph_nodes_vec", time.time() - start_time)
        return g_new
            
   

    def _update_merge_graph_edge_weigths_features(self, g, ntype, eids ):
        start_time = time.time()
        if "feat" not in self.coarsened_graph.nodes[ntype].data:
            return
        feat = self.coarsened_graph.nodes[ntype].data['feat']
        
        # 1) Build flat src/dst lists
        nodes_u , nodes_v = g.find_edges(eids)  
      
        # 2) mask out self‐pairs (if any)
        mask = nodes_u != nodes_v
        nodes_u, nodes_v = nodes_u[mask], nodes_v[mask]                                         # [E']

        # 3) pull features and compute costs
        f1 = feat[nodes_u]       # [E'×H]
        f2 = feat[nodes_v]       # [E'×H]
        
        cu = self.coarsened_graph.nodes[ntype].data["node_size"][nodes_u]
        cv = self.coarsened_graph.nodes[ntype].data["node_size"][nodes_v]        
       # g.edata[]
        mid = (f1* cu + f2* cv) / (cu + cv)   # [E'×H]
        
        costs =  torch.norm(mid - f1,  dim=1, p =1) + torch.norm(mid - f2,  dim=1, p=1) 
        g.edata["edge_weight_feat"][eids] =  costs
        if self.is_eval_metrics:
            self.feat_distances[ntype].append( costs)
        costs =  torch.norm(mid - f1,  dim=1, p =1) + torch.norm(mid - f2,  dim=1, p=1) 
            
            #costs = (torch.norm(mid - f1,  dim=1, p =1) + torch.norm(mid - f2,  dim=1, p=1))  / (self.minmax_ntype[ntype][2])  # [E']
       # print("_update_merge_graph_edge_weigths_features", time.time()- start_time)
        return  costs
    
    
    def _update_merge_graph_edge_weigts_neig_H_approx(self,costs, src_type, etype, node1_ids, node2_ids):

        for src_type_2, etype_2, dst_type_2 in self.coarsened_graph.canonical_etypes:
            if src_type != dst_type_2:
                continue
            feat1 = self.coarsened_graph.nodes[src_type].data["feat"][node1_ids]
            feat2 = self.coarsened_graph.nodes[src_type].data["feat"][node2_ids]
            
            c1 = self.coarsened_graph.nodes[src_type].data["node_size"][node1_ids]
            c2 = self.coarsened_graph.nodes[src_type].data["node_size"][node2_ids]
            if self.is_adj:
                deg = self._get_degree_with_adj(node1_ids, [], etype, True)
                d1 = deg[node1_ids]
                deg = self._get_degree_with_adj(node2_ids, [], etype, True)
                d2 = deg[node2_ids]
            #    d2 = deg[node2_ids]
            else:
                d1 = self.coarsened_graph.in_degrees(node1_ids,etype=etype_2)
                d2 = self.coarsened_graph.in_degrees(node2_ids,etype=etype_2)
            
            feat = (c1 * feat1 + c2 * feat2) / (c1 + c2 )
            
            diff1 = torch.norm( feat   / torch.sqrt(d1.unsqueeze(-1) + d2.unsqueeze(-1) + c1 + c2)  - feat1 /  torch.sqrt(d1.unsqueeze(-1) + c1)  , p=1, dim=1  )
            diff2 = torch.norm( feat   / torch.sqrt(d1.unsqueeze(-1) + d2.unsqueeze(-1) + c1 + c2)  - feat2 /  torch.sqrt(d2.unsqueeze(-1) + c2)  , p=1, dim=1  )
            cost_new = self.coarsened_graph.nodes[src_type].data[f"i{etype}"][node1_ids] * diff1 + self.coarsened_graph.nodes[src_type].data[f"i{etype}"][node2_ids] * diff2
            torch.add(costs, cost_new)
            
        
        
    def _update_merge_graph_edge_weigts_neig_H(self,costs, src_type, etype, etype_2 ,src_type_2,nodes_u_old, nodes_v_old):
        
            
            neigbors_u, nodes_u = self.coarsened_graph.in_edges(nodes_u_old, etype=etype_2)
            neigbors_v, nodes_v = self.coarsened_graph.in_edges(nodes_v_old, etype=etype_2)
            
            feat1 = self.coarsened_graph.nodes[src_type].data["feat"][nodes_u_old]
            feat2 = self.coarsened_graph.nodes[src_type].data["feat"][nodes_v_old]
            
            c1 = self.coarsened_graph.nodes[src_type].data["node_size"][nodes_u_old]
            c2 = self.coarsened_graph.nodes[src_type].data["node_size"][nodes_v_old]
            if self.is_adj:
                deg = self._get_degree_with_adj(nodes_u_old, nodes_v_old, etype)
                d1 = deg[nodes_u_old]
                d2 = deg[nodes_v_old]
            else:
                d1 = self.coarsened_graph.out_degrees(nodes_u_old,etype=etype)
                d2 = self.coarsened_graph.out_degrees(nodes_v_old,etype=etype)
            
            feat = (c1 * feat1 + c2 * feat2) / (c1 + c2 )
            
            adj = self.coarsened_graph.edges[etype_2].data["adj"]
            
            neighbors_u_extra_costs = torch.norm( feat * (adj[nodes_u_old]  + adj[nodes_v_old]).unsqueeze(1) / (torch.sqrt(d1.unsqueeze(1) + d2.unsqueeze(1) + c1 + c2))  - feat1 *  adj[nodes_u_old].unsqueeze(1) / (torch.sqrt(d1.unsqueeze(1) + c1)) , p=1, dim=1)
            neighbors_v_extra_costs = torch.norm( feat * (adj[nodes_u_old]  + adj[nodes_v_old]).unsqueeze(1)/ (torch.sqrt(d1.unsqueeze(1) + d2.unsqueeze(1) + c1 + c2))  - feat2 * adj[nodes_v_old].unsqueeze(1)/ (torch.sqrt(d2.unsqueeze(1) + c2)) , p=1, dim=1)
         
            if self.is_adj:
                
               # edges = self.coarsened_graph.edges(etype=etype_2)
                du = torch.zeros(neigbors_u.shape[0], device=self.device)
                du = du.index_add(0, neigbors_u, adj[neigbors_u])
                
                dv = torch.zeros(neigbors_v.shape[0], device=self.device)
                dv = dv.index_add(0, neigbors_v, adj[neigbors_v])
                
                # deg_t = self._get_degree_with_adj(neigbors_u, neigbors_v, etype_2)
                # print(deg_t)
                # du = deg_t[neigbors_u]
                # dv = deg_t[neigbors_v]
            else:
                du = self.coarsened_graph.out_degrees(neigbors_u, etype=etype_2 )
                dv = self.coarsened_graph.out_degrees(neigbors_v, etype=etype_2 )
            
            cu = self.coarsened_graph.nodes[src_type_2].data["node_size"][neigbors_u]
            cv = self.coarsened_graph.nodes[src_type_2].data["node_size"][neigbors_v]
            
            neighbors_u_extra_costs = neighbors_u_extra_costs[nodes_u] / torch.sqrt(du + cu.squeeze())
            neighbors_v_extra_costs = neighbors_v_extra_costs[nodes_v] / torch.sqrt(dv + cv.squeeze())
            
            
            neighbors_u_extra_costs = neighbors_u_extra_costs    
            neighbors_v_extra_costs = neighbors_v_extra_costs  
            #   total_cost = (cost_src + cost_dst) / 
            
            costs.index_add_(0, nodes_u, neighbors_u_extra_costs)
            costs.index_add_(0, nodes_v, neighbors_v_extra_costs)
            return costs

            

    def _update_merge_graph_edge_weights_H(self,g, ntype, eids):
        
        start_time = time.time()
        
        
        nodes_u_old , nodes_v_old = g.find_edges(eids)  
        costs = torch.zeros(len(nodes_u_old), device=self.device)
        
        for src_type, etype, dst_type in self.coarsened_graph.canonical_etypes:
            if src_type != ntype:
                continue
            
            
                
            
            merged_repr = self._create_h_via_cache_vec_fast_without_table(nodes_u_old, nodes_v_old, src_type, etype, self.coarsened_graph.nodes[src_type].data['node_size'])
            
            
           
            
            
            repr_u = self.coarsened_graph.nodes[src_type].data[f'h{etype}'][nodes_u_old]   # TODO: wrong !!!
            repr_v = self.coarsened_graph.nodes[src_type].data[f'h{etype}'][nodes_v_old]   

            cost_src =torch.norm(repr_u - merged_repr,  dim=1, p=1)  # [E]
            cost_dst = torch.norm(repr_v - merged_repr,  dim=1, p=1)  # [E]
            
                
             #   total_cost = self.zscore((cost_src + cost_dst) , self.means[etype] ,self.stds[etype] ) 
            costs =  cost_src + cost_dst 
            if self.is_eval_metrics:
                self.edge_distances[etype].append( costs)
            g.edata[f"edge_weight_{etype}"][eids] = costs
                #total_cost = (cost_src + cost_dst) / (self.minmax_etype[etype][2])               # [E]
           # costs += total_cost       
          #  self._update_merge_graph_edge_weigts_neig_H_approx(costs, src_type, etype, nodes_u_old, nodes_v_old)
            if self.is_neighboring_h:
                t = time.time()
                for src_type_2, etype_2, dst_type_2 in self.coarsened_graph.canonical_etypes:
                    if src_type != dst_type_2:
                        continue
        
                    g.edata[f"edge_neig_weight_{etype_2}"][eids] = self._update_merge_graph_edge_weigts_neig_H(costs, src_type, etype, etype_2 ,src_type_2, nodes_u_old, nodes_v_old)
                print("longer time", t - time.time())
            
            
        
      #  print("_update_merge_graph_edge_weights_H", time.time()- start_time)
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
            
            
            self.costs_features = dict()#self._update_merge_graph_edge_weigths_features(g_coar,ntype, self.edge_ids_need_recalc)
            
            self.costs_H = self._update_merge_graph_edge_weights_H(g_coar, ntype, self.edge_ids_need_recalc)
            
            # d_total = torch.zeros(g_coar.num_edges() ,device=self.device)
            # for src_type, etype, dst_type in self.coarsened_graph.canonical_etypes:
            #     if src_type == ntype:
            #         d_total += self.coarsened_graph.out_degrees(g_coar.edges()[0], etype=etype)
                
            if "edge_weight_feat" in g_coar.edata:
                score = self.zscore(g_coar.edata["edge_weight_feat"], ntype)
                if self.R:
                    score = score * self.R[ntype]
                if self.is_eval_metrics:
                    self.feat_scores[ntype].append(score)
                g_coar.edata["edge_weight"] = score
            else:
                g_coar.edata["edge_weight"] = torch.zeros(g_coar.num_edges(), device=self.device)
            
            for src_type, etype, dst_type in self.coarsened_graph.canonical_etypes:
                if src_type == ntype:
                  #  d_e = self.coarsened_graph.out_degrees(g_coar.edges()[0], etype=etype)
                    score =  self.zscore(g_coar.edata[f"edge_weight_{etype}"], etype)
                    if self.R:
                        score = score * self.R[etype]
                    if self.is_eval_metrics:
                        self.edge_scores[etype].append( score)
                    g_coar.edata["edge_weight"] += score# * d_e / d_total
                    if self.is_neighboring_h:
                        for src_type_2, etype_2, dst_type_2 in self.coarsened_graph.canonical_etypes:
                            if src_type != dst_type_2:
                                continue
                            score =  self.zscore(g_coar.edata[f"edge_neig_weight_{etype_2}"], etype_2, False)
                            if self.R:
                                score = score * self.R[etype]
                            g_coar.edata["edge_weight"] += score
                        
            #g_coar.edata["edge_weight"][self.edge_ids_need_recalc] = self.costs_features 
            #g_coar.edata["edge_weight"][self.edge_ids_need_recalc] += self.costs_H
            print("_update_merge_graph", time.time()- start_time)    
            return g_coar,   True
        else:
            print("_update_merge_graph: WARNING no more merge candidates", time.time()- start_time)    
            return g_coar,  False
        
    def _get_degree_with_adj(self, node1s, node2s, etype, is_out=True):
        if node2s == None:
            required = node1s
        else:
            required  = torch.cat([node1s, node2s])  # every key that must be present

        # 2) Grab degrees and caches in one go:
        adj = self.coarsened_graph.edges[etype].data["adj"]
        edges = self.coarsened_graph.edges(etype=etype)
        
        deg = torch.zeros(len(adj), dtype=adj.dtype, device=self.device)
        if is_out:
            edges_src = edges[0]
        else:
            edges_src = edges[1]
        
        deg.index_add_(0, edges_src, adj)
        return deg
        # 1) find which required keys are *not* yet present
        missing_mask = ~torch.isin(required, edges_src)          # -> tensor([False, False,  True,  True])
        missing_keys = required[missing_mask]               # -> tensor([2, 3])

        # 2) grow the two tensors
        
        aug_keys   = torch.cat((edges_src,   missing_keys))                      # [0, 0, 1, 2, 3]
        aug_values = torch.cat((adj, torch.zeros_like(missing_keys)))    # [2, 1, 2, 0, 0]
        
        
        uniq, inverse = torch.unique(aug_keys, return_inverse=True, sorted=True)
        #        uniq = tensor([0, 1])
        #     inverse = tensor([0, 0, 1])

        # ②  bucket-sum the values with index_add_
        deg = torch.zeros(len(uniq), dtype=adj.dtype, device=self.device)
        deg.index_add_(0, inverse, aug_values)
        
        
        return deg
        
      
    def _create_h_via_cache_vec_fast(self,  table,ntype, etype, cluster_sizes):
        cache = self.coarsened_graph.nodes[ntype].data[f's{etype}']
        
        # 1) Flatten your table into two 1-D lists of equal length L:
        pairs = [(u, v) for u, vs in table.items() for v in vs]
        node1s, node2s = zip(*pairs)
        node1s = torch.tensor(node1s, dtype=torch.long, device=self.device)
        node2s = torch.tensor(node2s, dtype=torch.long, device=self.device)


        if self.is_adj:
            deg = self._get_degree_with_adj(node1s, node2s, etype)
        else:    
            deg = self.coarsened_graph.out_degrees(etype=etype)
         
        
        deg1 = deg[node1s]  # shape (L,)
        deg2 = deg[node2s]            # shape (L,)

        su = cache[node1s]            # shape (L, D)
        sv = cache[node2s]            # shape (L, D)
        
        adj1 = self._get_adj(node1s, node2s, etype)
        adj2 = self._get_adj(node2s, node1s, etype)
        
        
        
        feat_u = self.coarsened_graph.nodes[ntype].data["feat"][node1s]
        feat_v = self.coarsened_graph.nodes[ntype].data["feat"][node2s]
        
        
        cu = cluster_sizes[node1s]
        cv = cluster_sizes[node2s]
        minus = torch.mul((adj2 / torch.sqrt(deg1 + cu.squeeze() )).unsqueeze(1), feat_u) + torch.mul((adj1 / torch.sqrt(deg2 + cv.squeeze() )).unsqueeze(1), feat_v) # + torch.matmul( (adj / torch.sqrt(deg2 + cv.squeeze() )), feat_v)
        #minus = feat_u*(g.edges[etype].data["adj"] [repeat_src] / torch.sqrt(du.unsqueeze(1) + cu ))  +  g.etype[etype].data["adj"] [repeat_dst] / torch.sqrt(dv + cv ) * feat_v
        

        # 3) Cluster‐size term (make sure cluster_sizes is a tensor):
        #csize = torch.tensor([cluster_sizes[i] for i in range(self.coarsened_graph.num_nodes())],
        #                    device=deg.device, dtype=deg.dtype)
        cuv = cluster_sizes[node1s] + cluster_sizes[node2s]  # shape (L,)

        # 4) Single vectorized compute of h for all L pairs:
        #    (we broadcast / unsqueeze cuv into the right D-dimensional form)
        
        feat = (feat_u*cu.unsqueeze(1) + feat_v*cv.unsqueeze(1)) / (cu.unsqueeze(1) + cv.unsqueeze(1))
        h_all = ((su + sv) - minus )/ torch.sqrt((deg1 + deg2 + cuv.squeeze())).unsqueeze(1 ) + feat * ( cuv / ((deg1 + deg2 + cuv.squeeze()))).unsqueeze(1)  #+   #)  # (L, D)

        return h_all
   
    def _create_h_via_cache_vec_fast_without_table(self,  node1s, node2s,ntype, etype, cluster_sizes):
        cache = self.coarsened_graph.nodes[ntype].data[f's{etype}']
        
        # 1) Flatten your table into two 1-D lists of equal length L:
        node1s = torch.tensor(node1s, dtype=torch.long, device=self.device)
        node2s = torch.tensor(node2s, dtype=torch.long, device=self.device)

        # 2) Grab degrees and caches in one go:
        if self.is_adj:
            deg = self._get_degree_with_adj(node1s, node2s, etype)
        else:
            deg = self.coarsened_graph.out_degrees(etype=etype)
        deg1 = deg[node1s]  # shape (L,)
        deg2 = deg[node2s]            # shape (L,)

        
        su = cache[node1s]            # shape (L, D)
        sv = cache[node2s]
        
        adj1 = self._get_adj(node1s, node2s, etype)
        adj2 = self._get_adj(node2s, node1s, etype)
        
        feat_u = self.coarsened_graph.nodes[ntype].data["feat"][node1s]
        feat_v = self.coarsened_graph.nodes[ntype].data["feat"][node2s]
        
        
        cu = cluster_sizes[node1s]
        cv = cluster_sizes[node2s]
        minus = torch.mul((adj2 / torch.sqrt(deg1 + cu.squeeze() )).unsqueeze(1), feat_u) + torch.mul((adj1 / torch.sqrt(deg2 + cv.squeeze() )).unsqueeze(1), feat_v) # + torch.matmul( (adj / torch.sqrt(deg2 + cv.squeeze() )), feat_v)
        #minus = feat_u*(g.edges[etype].data["adj"] [repeat_src] / torch.sqrt(du.unsqueeze(1) + cu ))  +  g.etype[etype].data["adj"] [repeat_dst] / torch.sqrt(dv + cv ) * feat_v
        

        # 3) Cluster‐size term (make sure cluster_sizes is a tensor):
        #csize = torch.tensor([cluster_sizes[i] for i in range(self.coarsened_graph.num_nodes())],
        #                    device=deg.device, dtype=deg.dtype)
        cuv = cluster_sizes[node1s] + cluster_sizes[node2s]  # shape (L,)

        # 4) Single vectorized compute of h for all L pairs:
        #    (we broadcast / unsqueeze cuv into the right D-dimensional form)
        feat = (feat_u*cu + feat_v*cv) / (cu + cv)
        h_all = ((su + sv) - minus )/ torch.sqrt((deg1 + deg2 + cuv.squeeze())).unsqueeze(1 ) + feat * ( cuv.squeeze() / ((deg1 + deg2 + cuv.squeeze()))).unsqueeze(1)  #+   #)  # (L, D)

        return h_all
    
    def _get_adj(self, nodes_u, nodes_v, etype):
        exists = self.coarsened_graph.has_edges_between(nodes_u, nodes_v, etype=etype)
        adj = torch.zeros(len(nodes_u), device=self.device)
        
                       # Get indices where the edge exists
        existing_edge_indices = exists.nonzero(as_tuple=True)[0]


        # Get edge IDs for the existing edges
        edge_ids = self.coarsened_graph.edge_ids(nodes_u[existing_edge_indices], nodes_v[existing_edge_indices], etype=etype)
        #g_new.nodes[node_type].data[f'i{etype}'][new_nodes] =   infl_uv
    
        edge_data = self.coarsened_graph.edges[etype].data['adj'][edge_ids]

        # Assign edge data to the result tensor
        adj[existing_edge_indices] = edge_data
        return adj
    
    
    
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
        for src_type, etype, dst_type in self.coarsened_graph.canonical_etypes:
            # ensure nested dict
            costs_dict.setdefault(src_type, {})[etype] = {}
            # compute all merged h representations in one go
            self.coarsened_graph.edges[etype].data["adj"] = torch.ones(self.coarsened_graph.num_edges(etype=etype), device=self.device)

            H_merged = self._create_h_via_cache_vec_fast(
                merge_list[src_type], src_type, etype,
                torch.ones(self.coarsened_graph.number_of_nodes(src_type), device=self.device)
            )  # [N_src, hidden]
            
            # flatten all (u,v) pairs same as above
            starts, ends = [], []
            for u, vs in merge_list[src_type].items():
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
            h1 = self.coarsened_graph.nodes[src_type].data[f"h{etype}"][node1_ids]  # [P, H]
            h2 = self.coarsened_graph.nodes[src_type].data[f"h{etype}"][node2_ids]  # [P, H]
             # build a dense [num_src, hidden] tensor
            #H_tensor =  torch.tensor([v for k,v in  H_merged.items()] , device=device)
            merged = H_merged                               # [P, H]
            
            
            
            
            # L1 costs
            cost = torch.norm(merged - h1, p=1, dim=1) + torch.norm(merged - h2, p=1, dim=1)
        #    cost = (h1 - merged).abs().sum(dim=1) + (h2 - merged).abs().sum(dim=1)  # [P]

            
                
            costs_dict[src_type][etype] = {
                "costs": cost,
                "index": torch.stack([node1_ids, node2_ids], dim=0)
            }
        return costs_dict
    def _approx_neighbors_h_costs(self, merge_list):
        costs_dict = {}
        for src_type, etype, dst_type in self.coarsened_graph.canonical_etypes:
            costs_dict.setdefault(src_type, {})[etype] = {}
            starts, ends = [], []
            for u, vs in merge_list[src_type].items():
                vs = [v for v in vs if v != u]
                if not vs:
                    continue
                starts.append(torch.full((len(vs),), u, dtype=torch.long, device=self.device))
                ends.append(torch.tensor(vs, dtype=torch.long, device=self.device))

            if not starts:
                continue

            node1_ids = torch.cat(starts)  # [P]
            node2_ids = torch.cat(ends)    # [P]
            for src_type_2, etype_2, dst_type_2 in self.coarsened_graph.canonical_etypes:
                if src_type != dst_type_2:
                    continue
                costs_dict[src_type].setdefault(etype, {})[etype_2] = {}
                feat1 = self.coarsened_graph.nodes[src_type].data["feat"][node1_ids]
                feat2 = self.coarsened_graph.nodes[src_type].data["feat"][node2_ids]
                
                c1 = self.coarsened_graph.nodes[src_type].data["node_size"][node1_ids]
                c2 = self.coarsened_graph.nodes[src_type].data["node_size"][node2_ids]
                
                d1 = self.coarsened_graph.in_degrees(node1_ids,etype=etype_2)
                d2 = self.coarsened_graph.in_degrees(node2_ids,etype=etype_2)
                # torch.ones(feat1.shape[0]) 
                feat = (c1 * feat1 + c2 * feat2) / (c1 + c2 )
                
                diff1 = torch.norm( feat   / torch.sqrt(d1.unsqueeze(-1) + d2.unsqueeze(-1) + c1 + c2)  - feat1 /  torch.sqrt(d1.unsqueeze(-1) + c1)  , p=1, dim=1  )
                diff2 = torch.norm( feat   / torch.sqrt(d1.unsqueeze(-1) + d2.unsqueeze(-1) + c1 + c2)  - feat2 /  torch.sqrt(d2.unsqueeze(-1) + c2)  , p=1, dim=1  )
                cost = self.coarsened_graph.nodes[src_type].data[f"i{etype}"][node1_ids] * diff1 + self.coarsened_graph.nodes[src_type].data[f"i{etype}"][node2_ids] * diff2
                costs_dict[src_type][etype][etype_2] = {
                "costs": cost,
                "index": torch.stack([node1_ids, node2_ids], dim=0)
                }
        return costs_dict
    
                
    def neighbors_very_expensive(self, node1_ids, node2_ids, etype, etype_2, src_type):
             #   costs_dict[src_type].setdefault(etype, {})[etype_2] = {}
        pairs = torch.stack((node1_ids, node2_ids), dim=1)
        costs_array = []
        for p1, p2 in pairs:
            neighbors,nodes  = self.coarsened_graph.in_edges(torch.tensor((p1,p2), device=self.device), etype=etype_2)
            mask1 = neighbors != p1
            mask2 = neighbors != p2
            mask = torch.logical_and(mask1, mask2)
            neighbors = neighbors[mask]
            cp1 = self.coarsened_graph.nodes[src_type].data["node_size"][p1]
            cp2 = self.coarsened_graph.nodes[src_type].data["node_size"][p2]
            feat_1 = self.coarsened_graph.nodes[src_type].data["feat"][p1]
            feat_2 = self.coarsened_graph.nodes[src_type].data["feat"][p2]
            d1 = self._get_degree_with_adj(p1, None,etype )[p1]
            d2 = self._get_degree_with_adj(p2, None,etype )[p2]
            feat_p = (cp1 * feat_1 + cp2 * feat_2) / (cp1 + cp2 )
            costs = 0
            for n in neighbors:
                h = self.coarsened_graph.nodes[src_type].data[f"h{etype}"][n]
                cn =  self.coarsened_graph.nodes[src_type].data["node_size"][n]
                dn = self._get_degree_with_adj(n, None,etype )[n]
                adj = self._get_adj(torch.tensor((n, n), device=self.device), torch.tensor((p1, p2), device=self.device), etype)
                #   adj_2_n = self._get_adj(torch.tensor(p2), torch.tensor(n), etype)
                feat_n = self.coarsened_graph.nodes[src_type].data["feat"][n]
                sn = self.coarsened_graph.nodes[src_type].data[f's{etype}'][n]
                h_prim = cn * feat_n / (cn + dn) 
                h_prim += sn / torch.sqrt(cn+dn)
                h_prim += ((adj[0] +adj[1]) * feat_p ) / torch.sqrt((dn + cn) * ( d1 + d2 + cp1 + cp2))
                h_prim -= adj[0] * feat_1 / torch.sqrt((dn + cn) * ( d1 + cp1 ))
                h_prim -= adj[1] * feat_2 / torch.sqrt((dn + cn) * (  d2 + cp2)) 
                costs += torch.norm(h_prim - h, p=1)
            costs_array.append(costs)
        
        
        return costs_array

    def neighbors_expensive(self, node1_ids, node2_ids, etype, etype_2, src_type):
             #   costs_dict[src_type].setdefault(etype, {})[etype_2] = {}
        pairs = torch.stack((node1_ids, node2_ids), dim=1)
        costs_array = []
        neighbors,nodes  = self.coarsened_graph.in_edges(torch.cat((node1_ids,node2_ids)), etype=etype_2)
        d1_wtf = self._get_degree_with_adj(node1_ids, None,etype )
        d2_wtf = self._get_degree_with_adj(node2_ids, None,etype )
            
        for p1, p2 in tqdm(pairs):
            d1 = d1_wtf[p1]
            d2 = d2_wtf[p2]
            mask1 = nodes == p1
            mask2 = nodes == p2
            mask = torch.logical_or(mask1, mask2)
            neighbors_p = neighbors[mask]
            mask1 = neighbors_p != p1
            mask2 = neighbors_p != p2
            mask = torch.logical_and(mask1, mask2)
            neighbors_p = neighbors_p[mask]
            if len(neighbors) == 0:
                continue
            cp1 = self.coarsened_graph.nodes[src_type].data["node_size"][p1]
            cp2 = self.coarsened_graph.nodes[src_type].data["node_size"][p2]
            feat_1 = self.coarsened_graph.nodes[src_type].data["feat"][p1]
            feat_2 = self.coarsened_graph.nodes[src_type].data["feat"][p2]
            feat_p = (cp1 * feat_1 + cp2 * feat_2) / (cp1 + cp2 )
            costs = 0
            
            
            
            
    
            h = self.coarsened_graph.nodes[src_type].data[f"h{etype}"][neighbors]
            cn =  self.coarsened_graph.nodes[src_type].data["node_size"][neighbors]
            dn = d1_wtf[neighbors].unsqueeze(1)
                
            adj = self._get_adj(torch.cat((neighbors, neighbors)), torch.cat((p1.repeat(len(neighbors)), p2.repeat(len(neighbors)))), etype)
                #   adj_2_n = self._get_adj(torch.tensor(p2), torch.tensor(n), etype)
            feat_n = self.coarsened_graph.nodes[src_type].data["feat"][neighbors]
            sn = self.coarsened_graph.nodes[src_type].data[f's{etype}'][neighbors]
            h_prim = cn * feat_n / (cn + dn) 
            h_prim += sn / torch.sqrt(cn+dn)
            h_prim += ((adj[:(len(neighbors))] +adj[(len(neighbors)):]).unsqueeze(1) * feat_p ) / torch.sqrt((dn + cn) * ( d1 + d2 + cp1 + cp2))
            h_prim -= (adj[:(len(neighbors))].unsqueeze(1) * feat_1) / torch.sqrt((dn + cn) * ( d1 + cp1 ))
            h_prim -= (adj[(len(neighbors)):].unsqueeze(1) * feat_2) / torch.sqrt((dn + cn) * (  d2 + cp2)) 
            costs += torch.norm(h_prim - h, p=1)
            costs_array.append(costs)
        
        
        return costs_array
    
        
    def create_mapped_tensor(self,a, b, c, fill_value=float('nan')):
        """
        Create a tensor d where d[i,j] = c[k] if a[i] == b[k] and b[k] is the j-th unique value in b
        
        Parameters:
        a: tensor-like, shape (n,)
        b: tensor-like, shape (m,) 
        c: tensor-like, shape (m,), same length as b
        fill_value: value to use for missing combinations (default: nan)
        
        Returns:
        d: tensor, shape (len(a), len(unique(b)))
        """
        a = torch.tensor(a, dtype=torch.float32, device=self.device)
        b = torch.tensor(b, dtype=torch.float32, device=self.device)
        c = torch.tensor(c, dtype=torch.float32, device=self.device)
        
        # Get unique values in b and their indices
        unique_b, inverse_indices = torch.unique(b, return_inverse=True)
        
        # Create output tensor
        d = torch.full((len(a), len(unique_b)), fill_value, dtype=torch.float32, device=self.device)
        
        # Create boolean mask: a[:, None] == b[None, :]
        # This gives us a matrix where entry (i,j) is True if a[i] == b[j]
        mask = a[:, None] == b[None, :]
        
        # For each position in a, find which positions in b match
        # Then map those to the corresponding unique value indices
        row_indices, col_indices_b = torch.where(mask)
        col_indices_unique = inverse_indices[col_indices_b]
        
        # Fill the tensor with corresponding c values
        d[row_indices, col_indices_unique] = c[col_indices_b]
        
        return d
    
    def index_matrix_with_nan_mask(self,d, lookup_matrix, fill_value=float('nan')):
        """
        Index into lookup_matrix using non-NaN values from d as indices
        If lookup_matrix is 2D, returns the whole row for each index
        
        Parameters:
        d: tensor with NaN and integer values, shape (H, W)
        lookup_matrix: 1D tensor or 2D matrix, shape (N,) or (N, K)
        fill_value: value to use where d has NaN
        
        Returns:
        result: tensor shape (H, W, K) if lookup_matrix is 2D, or (H, W) if 1D
        """
        if lookup_matrix.dim() == 1:
            # 1D lookup - same as before
            mask = ~torch.isnan(d)
            result = torch.full_like(d, fill_value, device=self.device)
            indices = d[mask].long()
            result[mask] = lookup_matrix[indices]
            return result
        
        else:
            # 2D lookup - return whole rows (memory efficient version)
            H, W = d.shape
            K = lookup_matrix.shape[1]
            
            # Create result tensor
            result = torch.full((H, W, K), fill_value, device=self.device)
            
            # Find non-NaN positions
            row_idx, col_idx = torch.where(~torch.isnan(d))
            
            # Get indices and lookup vectors
            indices = d[row_idx, col_idx].long()
            lookup_vectors = lookup_matrix[indices]  # Shape: (num_non_nan, K)
            
            # Assign directly using advanced indexing
            result[row_idx, col_idx] = lookup_vectors
            
            return result
    def neighbors_cheap(self, node1_ids, node2_ids, etype, etype_2, src_type):
             #   costs_dict[src_type].setdefault(etype, {})[etype_2] = {}
        pairs = torch.stack((node1_ids, node2_ids), dim=1)
        costs_array = []
        pairs1 = node1_ids
        pairs2 = node2_ids
     #   pairs1 = torch.cat((pairs1,torch.tensor([6])) )
        
        #for p1, p2 in pairs:
        all_neighbors,nodes  = self.coarsened_graph.in_edges(torch.unique(torch.cat((pairs1,pairs2))), etype=etype_2)
        neighbors_1 = self.create_mapped_tensor(pairs1, nodes, all_neighbors)
        neighbors_2 = self.create_mapped_tensor(pairs2, nodes, all_neighbors)
        h_1 = self.index_matrix_with_nan_mask(neighbors_1, self.coarsened_graph.nodes[src_type].data[f"h{etype}"])
        h_prim_1 = torch.zeros(h_1.shape, device=self.device)
        s_1 = self.index_matrix_with_nan_mask(neighbors_1, self.coarsened_graph.nodes[src_type].data[f"s{etype}"])
        f_1 = self.index_matrix_with_nan_mask(neighbors_1, self.coarsened_graph.nodes[src_type].data[f"feat"])
        s_2 = self.index_matrix_with_nan_mask(neighbors_2, self.coarsened_graph.nodes[src_type].data[f"s{etype}"])
        f_2 = self.index_matrix_with_nan_mask(neighbors_2, self.coarsened_graph.nodes[src_type].data[f"feat"])
        
        c1 = self.index_matrix_with_nan_mask(neighbors_1, self.coarsened_graph.nodes[src_type].data["node_size"])
        deg = self._get_degree_with_adj(all_neighbors, None,etype )
        deg1 = self.index_matrix_with_nan_mask(neighbors_1,deg).unsqueeze(2)
        h_prim_1 += (c1 / (c1 + deg1)) * f_1
        h_prim_1 += (c1 / torch.sqrt(c1 + deg1)) * s_1
        
        all_neighbors
        mask1 = neighbors != pairs1
        mask2 = neighbors != pairs2
        mask = torch.logical_and(mask1, mask2)
        neighbors = neighbors[mask]
#        if len(neighbors) == 0:

 #           continue
        cp1 = self.coarsened_graph.nodes[src_type].data["node_size"][p1]
        cp2 = self.coarsened_graph.nodes[src_type].data["node_size"][p2]
        feat_1 = self.coarsened_graph.nodes[src_type].data["feat"][p1]
        feat_2 = self.coarsened_graph.nodes[src_type].data["feat"][p2]
        d1 = self._get_degree_with_adj(p1, None,etype )[p1]
        d2 = self._get_degree_with_adj(p2, None,etype )[p2]
        feat_p = (cp1 * feat_1 + cp2 * feat_2) / (cp1 + cp2 )
        costs = 0
        
        
        
        

        h = self.coarsened_graph.nodes[src_type].data[f"h{etype}"][neighbors]
        cn =  self.coarsened_graph.nodes[src_type].data["node_size"][neighbors]
        dn = self._get_degree_with_adj(neighbors, None,etype )[neighbors].unsqueeze(1)
            
        adj = self._get_adj(torch.cat((neighbors, neighbors)), torch.cat((p1.repeat(len(neighbors)), p2.repeat(len(neighbors)))), etype)
            #   adj_2_n = self._get_adj(torch.tensor(p2), torch.tensor(n), etype)
        feat_n = self.coarsened_graph.nodes[src_type].data["feat"][neighbors]
        sn = self.coarsened_graph.nodes[src_type].data[f's{etype}'][neighbors]
        h_prim = cn * feat_n / (cn + dn) 
        h_prim += sn / torch.sqrt(cn+dn)
        h_prim += ((adj[:(len(neighbors))] +adj[(len(neighbors)):]).unsqueeze(1) * feat_p ) / torch.sqrt((dn + cn) * ( d1 + d2 + cp1 + cp2))
        h_prim -= (adj[:(len(neighbors))].unsqueeze(1) * feat_1) / torch.sqrt((dn + cn) * ( d1 + cp1 ))
        h_prim -= (adj[(len(neighbors)):].unsqueeze(1) * feat_2) / torch.sqrt((dn + cn) * (  d2 + cp2)) 
        costs += torch.norm(h_prim - h, p=1)
        costs_array.append(costs)
    
        
        return costs_array

    def vectorwise_isin(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a: (n, d), b: (m, d)
        # Return: (n,) boolean mask
        a_exp = a.unsqueeze(1)        # (n, 1, d)
        b_exp = b.unsqueeze(0)        # (1, m, d)
        matches = (a_exp == b_exp).all(dim=2)  # (n, m)
        return matches.any(dim=1)  
    
    def _neigbors_h_costs(self, merge_list):
        costs_dict = {}
        for src_type, etype, dst_type in self.coarsened_graph.canonical_etypes:
            #costs_dict.setdefault(src_type, {})[etype] = {}
            starts, ends = [], []
            for u, vs in merge_list[src_type].items():
                vs = [v for v in vs if v != u]
                if not vs:
                    continue
                starts.append(torch.full((len(vs),), u, dtype=torch.long, device=self.device))
                ends.append(torch.tensor(vs, dtype=torch.long, device=self.device))

            if not starts:
                continue

            node1_ids = torch.cat(starts)  # [P]
            node2_ids = torch.cat(ends)    # [P]
            for src_type_2, etype_2, dst_type_2 in self.coarsened_graph.canonical_etypes:
                
                if src_type != dst_type_2:
                    continue
                if src_type not in costs_dict:
                    costs_dict[src_type] = {}
                    
              #  costs_array = self.neighbors_very_expensive(node1_ids, node2_ids, etype, etype_2, src_type)
                costs_array_2 = self.neighbors_expensive(node1_ids, node2_ids, etype, etype_2, src_type)
               # costs_array_3 = self.neighbors_cheap(node1_ids, node2_ids, etype, etype_2, src_type)
                costs_dict[src_type][etype_2] = {
                    "costs": torch.tensor(costs_array_2, device=self.device),
                    "index": torch.stack([node1_ids, node2_ids], dim=0)
                }
                
                # Get unique source nodes from all pairs
                pairs_nodes = torch.cat([node1_ids, node2_ids])
                pairs = torch.stack((node1_ids, node2_ids), dim=1)
        
                unique_srcs = torch.unique(pairs_nodes)
                feat_u = self.coarsened_graph.nodes[src_type].data["feat"][node1_ids]
                
                # Get all out edges from coarsened graph
                all_srcs, all_dsts = self.coarsened_graph.in_edges(unique_srcs, etype=etype_2)
                max_degree = max(self.coarsened_graph.in_degrees(unique_srcs, etype=etype_2))
                h_prim = torch.zeros((len(pairs),feat_u.shape[1],max_degree))
                h = self.coarsened_graph.nodes[src_type].data[f"h{etype}"][all_srcs]
                
                
                
                # Repeat all_srcs and all_dsts for pairwise comparison
                # pairs.shape = [num_pairs, 2]
                p1s = pairs[:, 0]  # shape [num_pairs]
                p2s = pairs[:, 1]  # shape [num_pairs]
                

                # [E] = number of edges, [P] = number of pairs
                # shape: [P, E]
                mask_src_in_p1p2 = (all_srcs.unsqueeze(0) == p1s.unsqueeze(1)) | (all_srcs.unsqueeze(0) == p2s.unsqueeze(1))

                # shape: [P, E] - keep dsts not equal to p1 or p2
                mask_dst_not_p1 = all_dsts.unsqueeze(0) != p1s.unsqueeze(1)
                mask_dst_not_p2 = all_dsts.unsqueeze(0) != p2s.unsqueeze(1)
                mask_dst_not_uv = mask_dst_not_p1 & mask_dst_not_p2

                # Combine the two masks to get the final filter
                final_mask = mask_src_in_p1p2 & mask_dst_not_uv  # shape [P, E]
                self.coarsened_graph.nodes[src_type].data[f"s{etype}"]
                
                repeat_dst = all_dsts.repeat(len(p1s), 1)
                
                # Collect neighbors per pair
                h = torch.where(final_mask.unsqueeze(2), self.coarsened_graph.nodes[src_type].data[f"h{etype}"][repeat_dst], False )
                
                #h = self.coarsened_graph.nodes[src_type].data[f"h{etype}"][neighbors_per_pair]
                

                
                
                
               
                
                feat_u = self.coarsened_graph.nodes[src_type].data["feat"][node1_ids]
                feat_v = self.coarsened_graph.nodes[src_type].data["feat"][node2_ids]
                
                c1 = self.coarsened_graph.nodes[src_type].data["node_size"][node1_ids]
                c2 = self.coarsened_graph.nodes[src_type].data["node_size"][node2_ids]
                
                d1 = self.coarsened_graph.out_degrees(node1_ids,etype=etype)
                d2 = self.coarsened_graph.out_degrees(node2_ids,etype=etype)
                
                feat_uv = (c1 * feat_u + c2 * feat_v) / (c1 + c2 )
                
                
                
                
                deg = self._get_degree_with_adj(node1_ids, node2_ids,etype )
                d_uv = deg[node1_ids] + deg[node2_ids]
                c_uv = c1+ c2
                nodes_u,neigbors_u  = self.coarsened_graph.in_edges(node1_ids, etype=etype_2)
                nodes_v,neigbors_v  = self.coarsened_graph.in_edges(node2_ids, etype=etype_2)
                
                # a_pairs = torch.stack((node1_ids, node2_ids), dim=1)  # shape [N, 2]
                # b_pairs = torch.stack((nodes_u, neigbors_u), dim=1)  # shape [M, 2]
                
                # # Find which b_pairs are in a_pairs
                # mask = self.vectorwise_isin(b_pairs, a_pairs) == False

                # # Apply mask
                # nodes_u = nodes_u[mask]
                # neigbors_u = neigbors_u[mask]      
                
                # b_pairs = torch.stack((nodes_v, neigbors_v), dim=1)  # shape [M, 2]

                # # Find which b_pairs are in a_pairs
                # mask = self.vectorwise_isin(b_pairs, a_pairs) == False

                # # Apply mask
                # nodes_v = nodes_v[mask]
                # neigbors_v = neigbors_v[mask]        
          
        
                nodes_uv = torch.cat([nodes_u, nodes_v])
                neighbors = torch.cat([neigbors_u, neigbors_v])
                hi_neigh = self.coarsened_graph.nodes[src_type].data[f'h{etype}'][neighbors] # TODO etype?
                
                feat_neigh = self.coarsened_graph.nodes[src_type].data[f'feat'][neighbors]
                
               # c_uv =   self.coarsened_graph.nodes[src_type].data[f'node_size'][nodes_uv]
                c_neigh = self.coarsened_graph.nodes[src_type].data[f'node_size'][neighbors]
                 
                adj1_u_nei_u = self._get_adj(neigbors_u, nodes_u, etype)
                adj1_v_nei_v = self._get_adj(neigbors_v, nodes_v, etype)
        
                adj1_u_nei_u = self._get_adj(nodes_u, neigbors_u, etype)
                adj1_v_nei_v = self._get_adj(nodes_v, neigbors_v, etype)
        
                # adj1_u_nei_v = self._get_adj(nodes_u, neigbors_v, etype)
                # adj1_v_nei_u = self._get_adj(nodes_v, neigbors_u, etype)
        
                
                s_neigh = self.coarsened_graph.nodes[src_type].data[f"s{etype}"][neighbors]
                
                d_neigh = self._get_degree_with_adj(neighbors, None,etype )
                deg_u = self._get_degree_with_adj(nodes_u, None,etype )
                #deg_u.repeat()
                deg_v = self._get_degree_with_adj(nodes_v, None,etype )
                
               # feat_u = self.coarsened_graph.nodes[src_type].data["feat"][nodes_u]
                # feat_v = self.coarsened_graph.nodes[src_type].data["feat"][nodes_v]
                
               # d_v = self._get_degree_with_adj(nodes_v, None,etype )
                
                hi_prim = c_neigh / (d_neigh[neighbors].unsqueeze(1) + c_neigh )  * feat_neigh
                hi_prim += (1 / (torch.sqrt(d_neigh[neighbors].unsqueeze(1)  + c_neigh )))*s_neigh
                
                hi_prim[:len(nodes_u)] += ((adj1_u_nei_u ) * feat_uv[nodes_u].squeeze()).unsqueeze(1)  / torch.sqrt(((d_neigh[neigbors_u].unsqueeze(1) + c_neigh[neigbors_u]) *  (d_uv[nodes_u].unsqueeze(1)   + c_uv[nodes_u] )).squeeze()  ).unsqueeze(1)
                hi_prim[len(nodes_u):] += ((adj1_v_nei_v ) * feat_uv[nodes_v].squeeze()).unsqueeze(1)/ torch.sqrt(((d_neigh[neigbors_v].unsqueeze(1) + c_neigh[neigbors_v]) *  (d_uv[nodes_v].unsqueeze(1)   + c_uv[nodes_v] )).squeeze()  ).unsqueeze(1)
                
               # hi_prim = 
                
                
                hi_prim[:len(nodes_u)] -= (adj1_u_nei_u * feat_v[neigbors_v].squeeze()).unsqueeze(1) / torch.sqrt((d_neigh[neigbors_u].unsqueeze(1) + c_neigh[neigbors_u]) * (deg_u[nodes_u].unsqueeze(1) + c1[nodes_u]))
                hi_prim[len(nodes_u):] -= (adj1_v_nei_v * feat_u[nodes_v].squeeze()).unsqueeze(1) / torch.sqrt((d_neigh[neigbors_v].unsqueeze(1) + c_neigh[neigbors_v]) * (deg_v[nodes_v].unsqueeze(1) + c2[nodes_v]))
                
                feat_u = self.coarsened_graph.nodes[src_type].data["feat"][nodes_u]
                
                hi_prim[:len(nodes_u)] -= (adj1_u_nei_v * feat_v[nodes_u].squeeze()).unsqueeze(1) / torch.sqrt((d_neigh[neigbors_u].unsqueeze(1) + c_neigh[neigbors_u]) * (deg_u[nodes_u].unsqueeze(1) + c1[nodes_u]))
                hi_prim[len(nodes_u):] -= (adj1_v_nei_u * feat_v[nodes_v].squeeze()).unsqueeze(1) / torch.sqrt((d_neigh[neigbors_v].unsqueeze(1) + c_neigh[neigbors_v]) * (deg_v[nodes_v].unsqueeze(1) + c2[nodes_v]))
                
                
                
                distances = torch.norm(hi_prim - hi_neigh, p=1, dim=1)
                distances_v = distances[len(nodes_u):]
                distances_u = distances[:len(nodes_u)]
                
                
                for n1, n2 in zip(node1_ids, node2_ids):
                    cost = 0
                    for nu, nei_u in zip(nodes_u,neigbors_u):
                        if nu!=n1 and nei_u == n2:
                            cost += distances_u[u]
                    for nv, nei_v in zip(nodes_v,neigbors_v):
                        if nv!=n2 and nei_v == n1:
                            cost += distances_u[u]
                        
                    print(cost)
                
                
                mask_1 = neigbors_u[node1_ids] != node2_ids
                # Initialize mask with False
                mask = torch.zeros_like(node2_ids, dtype=torch.bool)

                # Find valid indices (i.e., node1[i] < len(nei))
                valid = node2_ids < len(neigbors_v)

                # Apply only on valid positions
                mask[valid] = neigbors_v[node2_ids[valid]] != node1_ids[valid]
                mask
                #mask_2 = neigbors_u != node1_ids[nodes_u]
                distances_u = torch.where(mask_1, distances_u[node1_ids], torch.tensor(0))
                distances_v = torch.where(mask, distances_v[node2_ids[valid]], torch.tensor(0))
                costs = torch.zeros(len(node1_ids), dtype=distances.dtype, device=self.device)
                costs = distances
                #costs.index_add_(0, nodes_u, distances[nodes_u] )
                #costs.index_add_(0, nodes_v, distances[nodes_v] )
                
                #h_u_prim += (adj1_u_nei[nodes_uv] + adj1_v_nei[nodes_uv]) / 
                #h_u_prim -= adj1_u_nei[neighbors] / torch.sqrt(1)                 
                #edge_ids_u = self.coarsened_graph.edge_ids(neigbors_u, nodes_u, etype=etype_2)
                edge_ids_v = self.coarsened_graph.edge_ids(neigbors_v, nodes_v, etype=etype_2)
                
                adj_u = self.coarsened_graph.edges[etype_2].data["adj"][edge_ids_u]
                adj_v = self.coarsened_graph.edges[etype_2].data["adj"][edge_ids_v]
                
                du = self.coarsened_graph.out_degrees(neigbors_u, etype=etype_2 )
                dv = self.coarsened_graph.out_degrees(neigbors_v, etype=etype_2 )
                
                cu = self.coarsened_graph.nodes[src_type_2].data["node_size"][neigbors_u]
                cv = self.coarsened_graph.nodes[src_type_2].data["node_size"][neigbors_v]
                
                neighbors_u_extra_costs = neighbors_u_extra_costs[nodes_u]/ torch.sqrt(du + cu.squeeze())
                neighbors_v_extra_costs = neighbors_v_extra_costs[nodes_v] / torch.sqrt(dv + cv.squeeze())
                
                
                cost = torch.zeros(node1_ids.shape[0], dtype=torch.float, device=self.device)
                cost = cost.index_add(0, nodes_u, neighbors_u_extra_costs)
                cost = cost.index_add(0, nodes_v, neighbors_v_extra_costs)
                costs_dict[src_type][etype_2] = {
                "costs": cost,
                "index": torch.stack([node1_ids, node2_ids], dim=0)
                }
                
                
        return costs_dict
    
    
    def percentile_score(self, x, *, dim=None):
        """
        Return the percentile score of each element in `x`
        along `dim` (default: entire flattened tensor).

        - Each score is in [0, 100].
        - NaNs are ignored but keep their position (score = NaN).
        """
        if dim is None:
            x_flat = x.flatten()
            order = torch.argsort(x_flat, stable=True)         # ranks
            ranks = torch.empty_like(order, dtype=torch.float,device=self.device)
            ranks[order] = torch.linspace(1, len(x_flat), steps=len(x_flat), device=self.device)
            pct = 100.0 * (ranks - 0.5) / len(x_flat)          # Hazen definition
            pct[x_flat.isnan()] = float('nan')                 # keep NaNs
            return pct.view_as(x)
        else:
            # Work per-dimension with a little reshaping magic
            orig_shape = x.shape
            x_t = x.transpose(0, dim).reshape(x.size(dim), -1)
            pct_t = self.percentile_score(x_t, dim=0)
            return pct_t.reshape(orig_shape).transpose(0, dim)

    def zscore(self,x: torch.Tensor, type,update_type=True,dim=None,  eps: float = 1e-8) -> torch.Tensor:
        """
        Return the z-score–normalised version of `x`.

        Parameters
        ----------
        x   : torch.Tensor
            Input matrix or higher-rank tensor.
        dim : int | tuple[int] | None, default None
            Dimension(s) over which to compute mean and std.
            • None   → use the whole tensor (global z-score)  
            • 0      → column-wise (each feature independently)  
            • 1      → row-wise, etc.
        eps : float, default 1e-8
            Small constant to avoid division by zero when a std is 0.

        Returns
        -------
        torch.Tensor
            Tensor of same shape as `x`, standardised so that
            (approximately) mean = 0 and std = 1 along `dim`.
        """
        if update_type:
            self.means[type] = x.mean(dim=dim, keepdim=True)
            self.stds[type]  = x.std(dim=dim, unbiased=False, keepdim=True)
        
        return (x - self.means[type]) / (self.stds[type] + eps)

    
    def min_max_scaler(self,x, eps=1e-13, dim=None):
        xmin = x.min(dim=0, keepdim=True)[0]
        xmax = x.max(dim=0, keepdim=True)[0]
        x_std = (x - xmin) / (xmax - xmin + eps)
        return x_std * (xmax - xmax) + xmin


                    
    def _costs_of_merges(self, merge_list):
        start_time = time.time()
        

        self.init_costs_dict_features = dict()  # self._feature_costs( merge_list)
        
        self.init_costs_dict_etype = self._h_costs( merge_list)    
       # self.init_something = self._approx_neighbors_h_costs(merge_list)
        if self.is_eval_metrics:
            for ntype in self.coarsened_graph.ntypes:
                self.feat_distances[ntype] = []
                self.feat_distances[ntype].append(self.init_costs_dict_features[ntype]["costs"])
                for src_type,etype,_ in self.coarsened_graph.canonical_etypes:
                    if src_type == ntype:
                        self.edge_distances[etype] = []
                        self.edge_distances[etype].append(self.init_costs_dict_etype[ntype][etype]["costs"])
        if self.is_neighboring_h:
            self.neighors_extra_cost = self._neigbors_h_costs(merge_list)
        else:
            self.neighors_extra_cost = None
        #self.init_costs_dict, self.init_index_dict = self._add_costs(self.init_costs_dict_features, self.init_costs_dict_etype, self.neighors_extra_cost)
       
        print("_costs_of_merges", time.time() - start_time)
        
        return self.init_costs_dict_features, self.init_costs_dict_etype, self.neighors_extra_cost
            
            
            
    def _get_master_mapping_tensor(self, mappings, ntype):
        master_mapping = dict()
        nodes_orig = self.original_graph.nodes(ntype)
        nodes = self.original_graph.nodes(ntype)
        for mapping in mappings:
            nodes = mapping[nodes]
        for i in range(len(nodes)):
            master_mapping[nodes_orig[i].item()] = nodes[i].item()
        
        return master_mapping    


       
    
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
        
        self.init_costs_dict_features, self.init_costs_dict_etype, self.neighors_extra_cost = self._costs_of_merges(self.init_neighbors)
        self._init_merge_graph(self.init_costs_dict_features, self.init_costs_dict_etype, self.neighors_extra_cost)
        
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
    tester = TestHomo()
    g = tester.g 
    tester.run_test(HeteroCoarsener(None,g, 0.5, num_nearest_per_etype=2, num_nearest_neighbors=2,pairs_per_level=3, is_neighboring_h=False, is_adj=True,device="cpu"))
    dataset = Citeseer() 
    original_graph = dataset.load_graph()

    #original_graph = create_test_graph()
    coarsener =  HeteroCoarsener(None,original_graph, 0.1, num_nearest_per_etype=10, num_nearest_neighbors=10,pairs_per_level=3, is_neighboring_h=True, is_adj=True) # 
    coarsener.init_step()
    for i in range(3):
        print("--------- step: " , i , "---------" )
        coarsener.iteration_step()
       # coarsener.get_mapping("author")