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
device = "cuda"# torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class HeteroCoarsener(GraphSummarizer):
    def __init__(self, dataset: Dataset, original_graph: dgl.DGLGraph, r: float, pairs_per_level: int = 10, 
                 num_nearest_neighbors: int = 10, num_nearest_per_etype:int = 10, filename = "dblp", R=None
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
        self.device = "cuda"#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.filename = filename
        assert (r > 0.0) and (r <= 1.0)
        self.r = r
        self.pca_components = 3
        self.original_graph = original_graph.to(device)
        self.dataset = dataset
        self.num_nearest_neighbors = num_nearest_neighbors
        self.num_nearest_per_etype = num_nearest_per_etype
        self.pairs_per_level = pairs_per_level
        self.coarsened_graph = original_graph.clone()
        self.coarsened_graph = self.coarsened_graph.to(device)
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
            deg_out = torch.tensor(self.node_degrees[etype]['out'], device=device) + 1.0
            deg_in  = torch.tensor(self.node_degrees[etype]['in'], device=device)  + 1.0
            inv_sqrt_out = torch.rsqrt(deg_out)
            inv_sqrt_in  = torch.rsqrt(deg_in)

            # Load features or use scalar 1
            if has_feat:
                feats = g.nodes[dst_type].data['feat'].to(device)
                feat_dim = feats.shape[1]
            else:
                # treat feature as scalar 1
                feat_dim = 1

            # Extract all edges of this type
            u, v = g.edges(etype=(src_type, etype, dst_type))
            u = u.to(device)
            v = v.to(device)

            # Gather destination feats & normalize
            if has_feat:
                feat_v = feats[v]                              # [E, D]
            else:
                feat_v = torch.ones((v.shape[0], 1), device=device)
            s_e = feat_v * inv_sqrt_in[v].unsqueeze(-1)       # [E, D]

            # Scatter-add to compute S at source nodes
            n_src = g.num_nodes(src_type)
            S_tensor = torch.zeros((n_src, feat_dim), device=device)
            S_tensor = S_tensor.index_add(0, u, s_e)

            # Compute H = D_out^{-1/2} * S
            H_tensor = inv_sqrt_out.unsqueeze(-1) * S_tensor

            # Store in coarsened_graph
            self.coarsened_graph.nodes[src_type].data[f's{etype}'] = S_tensor
            self.coarsened_graph.nodes[src_type].data[f'h{etype}'] = H_tensor
            self.coarsened_graph.nodes[src_type].data['node_size']  = torch.ones((n_src, 1), device=device)

        print("_create_h_spatial_rgcn", time.time() - start_time)
        
   

    
    def _init_merge_graph(self, costs, edges):
       
        start_time = time.time()
        self.merge_graphs =dict()
        for ntype in self.coarsened_graph.ntypes:
            self.merge_graphs [ntype] = dgl.graph(([], []), num_nodes=self.coarsened_graph.number_of_nodes(ntype=ntype), device=device)
            
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
        if H.size(1) > self.pca_components:
            # center and normalize
            mean = H.mean(dim=0, keepdim=True)
            std = H.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-4)
            Hn = (H - mean) / std
            # low rank PCA
            U, S, V = torch.pca_lowrank(Hn, q=self.pca_components)
            H = Hn @ V[:, :self.pca_components]

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
    
    


    def _merge_nodes_fast(self,g, node_type, node_pairs):
        """
        Merge multiple node_pairs in a single pass on GPU.
        """
        t0 = time.time()
        # 1) Build a new mapping: for every original node, what supernode does it go to?
        num_old = g.num_nodes(node_type)
        # start by mapping each node to itself
        mapping = torch.arange(num_old, device=g.device)
        # assign new supernode IDs: old nodes 0..num_old-1, merges num_old..num_old+P-1
        P = len(node_pairs)
        super_ids = torch.arange(num_old, num_old + P, device=g.device)
        # for each pair i, assign both nodes -> super_ids[i]
        nodes_u = torch.tensor([u for u,_ in node_pairs], device=g.device)
        nodes_v = torch.tensor([v for _,v in node_pairs], device=g.device)
        mapping = mapping.clone()
        mapping[nodes_u] = super_ids
        mapping[nodes_v] = super_ids

        # 2) Remap every edge to the new supernode IDs and coalesce duplicates.
        #    We'll do this per canonical etype.
        new_data = {}  # per-(etype, ntype) store new features
        new_edges = {}
        for srctype, etype, dsttype in g.canonical_etypes:
            u, v = g.edges(etype=(srctype, etype, dsttype))
            if srctype == node_type:
                u = mapping[u]
            if dsttype == node_type:
                v = mapping[v]

            # coalesce multiple edges into one: use DGL's to_simple
            # we'll build a tiny 2-node graph and then reattach features later
            subg = dgl.graph((u, v), num_nodes=num_old + P, device=g.device)
            subg = dgl.to_simple(subg, return_counts='count')  # merges parallel edges
            new_edges[(srctype, etype, dsttype)] = (subg.edges()[0], subg.edges()[1])

            # Aggregate features if this is a self‐loop edge type stored at the nodes
            # e.g. s_etype, h_etype for node‐type self‐loops
            if srctype ==  node_type:
                 
                # sum up the old s_etype features
                old_s = g.nodes[node_type].data[f's{etype}']
                sum_s = scatter_add(old_s[mapping[:num_old]], mapping[:num_old], dim=0, dim_size=num_old+P)
                new_s = sum_s
                # new node_size likewise
                old_size = g.nodes[node_type].data['node_size']
                sum_size = scatter_add(old_size[:num_old], mapping[:num_old], dim=0, dim_size=num_old+P)
                new_data[(node_type, f'node_size')] = sum_size
                # compute new h = s / sqrt(degree + size)
                deg = subg.out_degrees().float()
                new_data[(node_type, f's{etype}')] = new_s
                new_data[(node_type, f'h{etype}')] = new_s / torch.sqrt(deg + sum_size)

        # 3) Build the final merged graph
        #    Gather new edge lists and build heterograph
        data_dict = {}
        for c_etype, (uu, vv) in new_edges.items():
            data_dict[c_etype] = (uu.long(), vv.long())

        new_g = dgl.heterograph(data_dict, num_nodes_dict={nt: (num_old + P if nt==node_type else g.num_nodes(nt))
                                                        for nt in g.ntypes},
                                device=g.device)

        # Copy over aggregated node features
        for (nt, key), val in new_data.items():
            new_g.nodes[nt].data[key] = val

        # Also handle any “feat” field by size-weighted averaging:
        if 'feat' in g.nodes[node_type].data:
            feats = g.nodes[node_type].data['feat']
            sizes = new_data[(node_type, 'node_size')]
            # weighted sum then divide
            weighted = feats[:num_old] * g.nodes[node_type].data['node_size'][:num_old].unsqueeze(-1)
            sum_weighted = scatter_add(weighted, mapping[:num_old], dim=0, dim_size=num_old+P)
            new_g.nodes[node_type].data['feat'] = sum_weighted / sizes.unsqueeze(-1)

        print("merge_nodes_gpu", time.time() - t0)
        # return both new graph and mapping (if you still need it)
        return new_g, mapping

    
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
                if src_type != node_type and dst_type != node_type:
                    continue
                edges_original = g.edges(etype=etype)
                    
                if src_type == node_type:
                    
                    mask_node1 =  edges_original[0] == (mapping[node1]).item()
                    mask_node2 =  edges_original[0] == mapping[node2].item()
                    mask = torch.logical_or(mask_node1, mask_node2 )
                    edges_dst = torch.unique(edges_original[1][mask] )
                    edges_src = torch.full(edges_dst.shape,new_node_id, device=device )
                    g.add_edges(edges_src, edges_dst, etype=(src_type, etype,dst_type))
                    suv = g.nodes[node_type].data[f's{etype}'][mapping[node1]] + g.nodes[node_type].data[f's{etype}'][mapping[node2]]
                    cuv = g.nodes[node_type].data["node_size"][mapping[node1]] + g.nodes[node_type].data["node_size"][mapping[node2]]
                    g.nodes[node_type].data["node_size"][new_node_id] = cuv
                    duv = g.out_degrees(new_node_id, etype=etype) # TODO
                    g.nodes[node_type].data[f's{etype}'][new_node_id] = suv  
                    g.nodes[node_type].data[f'h{etype}'][new_node_id] =  suv / torch.sqrt(duv + cuv)
                
                if dst_type == src_type:
                    mask_node1 =  edges_original[1] == (mapping[node1]).item()
                    mask_node2 =  edges_original[1] == mapping[node2].item()
                    mask = torch.logical_or(mask_node1, mask_node2)
                    edges_src = torch.unique(edges_original[0][mask])
                    edges_dst = torch.full(edges_src.shape,new_node_id ,device=device)
                    g.add_edges(edges_src, edges_dst, etype=(src_type, etype,dst_type))
                
                elif dst_type == node_type:
                    mask_node1 =  edges_original[1] == (mapping[node1]).item()
                    mask_node2 =  edges_original[1] == mapping[node2].item()
                    mask = torch.logical_or(mask_node1, mask_node2)
                    edges_src = torch.unique(edges_original[0][mask])
                    edges_dst = torch.full(edges_src.shape,new_node_id , device=device)
                    g.add_edges(edges_src, edges_dst, etype=(src_type, etype,dst_type))
                
                
                
            
            if "feat" in g.nodes[node_type].data:
                old_feats = g.nodes[node_type].data["feat"] 
                cu = g.nodes[node_type].data["node_size"][mapping[node1]]
                cv = g.nodes[node_type].data["node_size"][mapping[node2]]
                g.nodes[node_type].data["feat"][new_node_id] = (old_feats[mapping[node1]] * cu  + old_feats[mapping[node2]] * cv ) / (cu + cv)
            
                
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
        
        g.edata["needs_check"] = torch.zeros(g.num_edges(), dtype=torch.bool, device=device) 
        for node1, node2 in node_pairs: #tqdm(, "merge nodes"):
            g.add_nodes(1)
            new_node_id =  g.num_nodes() -1
            
            edges_original = g.edges()
            mask_node1 =  torch.where(edges_original[0] == mapping[node1], True, False)
            mask_node2 =  torch.where(edges_original[0] == mapping[node2], True, False)
            mask = torch.logical_or(mask_node1, mask_node2)
            # TODO: dst nneds to be checked maybe too
            edges_dst = torch.unique(edges_original[1][mask])
            edges_src = torch.full(edges_dst.shape,new_node_id,device=device )
            g.add_edges(edges_src, edges_dst)
            edge_ids = g.edge_ids(edges_src, edges_dst)
            g.edata["needs_check"][edge_ids] = True
            
            mask_node1 =  torch.where(edges_original[1] == mapping[node1], True, False)
            mask_node2 =  torch.where(edges_original[1] == mapping[node2], True, False)
            mask = torch.logical_or(mask_node1, mask_node2)
            edges_dst = torch.unique(edges_original[0][mask])
            edges_src = torch.full(edges_dst.shape,new_node_id,device=device )
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
        
        cu = self.coarsened_graph.nodes[ntype].data["node_size"][src]
        cv = self.coarsened_graph.nodes[ntype].data["node_size"][dst]        
        
        mid = (f1* cu + f2* cv) / (cu + cv)   # [E'×H]
        if self.R:
            costs = (torch.norm(mid - f1,  dim=1, p =1) + torch.norm(mid - f2,  dim=1, p=1))  / (self.R[ntype])
        else:
            costs = (torch.norm(mid - f1,  dim=1, p =1) + torch.norm(mid - f2,  dim=1, p=1))  / (self.minmax_ntype[ntype][2])  # [E']
        print("_update_merge_graph_edge_weigths_features", time.time()- start_time)
        return  costs
        
        
   

    def _update_merge_graph_edge_weights_H(self,g, ntype, eids):
        
        start_time = time.time()
        
        
        src_nodes , dst_nodes = g.find_edges(eids)  
        candidates = dict()
        
        # 1) a value of 1 for each edge
        values = torch.ones(src_nodes.size(0), device=src_nodes.device)

        # 2) stack i,j indices
        indices = torch.stack([src_nodes, dst_nodes], dim=0)

        # move once
        src_cpu = src_nodes.to("cpu").numpy()
        dst_cpu = dst_nodes.to("cpu").numpy()

        candidates = {}
        for s, d in zip(src_cpu, dst_cpu):
            # this is now pure numpy ints – no per-element .item()
            candidates.setdefault(int(s), set()).add(int(d))
        costs = torch.zeros(len(src_nodes), device=device)
        
        for src_type, etype, dst_type in self.coarsened_graph.canonical_etypes:
            if src_type != ntype:
                continue
            
            merged_repr = self._create_h_via_cache_vec_fast(candidates, src_type, etype, self.coarsened_graph.nodes[src_type].data['node_size'])
            
            #merged_repr = torch.cat([H_merged[n] for n in candidates.keys()], dim=0)  

            src_repr = self.coarsened_graph.nodes[src_type].data[f'h{etype}'][src_nodes]   # TODO: wrong !!!
            dst_repr = self.coarsened_graph.nodes[src_type].data[f'h{etype}'][dst_nodes]   

            cost_src =torch.norm(src_repr - merged_repr,  dim=1, p=1)  # [E]
            cost_dst = torch.norm(dst_repr - merged_repr,  dim=1, p=1)  # [E]
            if self.R:
                total_cost = (cost_src + cost_dst) / (self.R[etype])
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
            size = data["node_size"].to(device)   # [N]

            # build flat list of all valid (u,v) pairs
            starts, ends = [], []
            for u, vs in merge_list[ntype].items():
                # filter out self‐merges if any
                vs = [v for v in vs if v != u]
                if not vs:
                    continue
                starts.append(torch.full((len(vs),), u, dtype=torch.long, device=device))
                ends.append(torch.tensor(vs, dtype=torch.long, device=device))

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
                torch.ones(self.coarsened_graph.number_of_nodes(src), device=device)
            )  # [N_src, hidden]

            # flatten all (u,v) pairs same as above
            starts, ends = [], []
            for u, vs in merge_list[src].items():
                vs = [v for v in vs if v != u]
                if not vs:
                    continue
                starts.append(torch.full((len(vs),), u, dtype=torch.long, device=device))
                ends.append(torch.tensor(vs, dtype=torch.long, device=device))

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
                
                norm = cost / (self.R.get(ntype, R))
            
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
                    norm = cost / self.R.get(etype, R)
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
        



if __name__ == "__main__":
    tester = TestHomo()
    g = tester.g 
    tester.run_test(HeteroCoarsener(None,g, 0.5, num_nearest_per_etype=2, num_nearest_neighbors=2,pairs_per_level=30))
    dataset = DBLP() 
    original_graph = dataset.load_graph()

    #original_graph = create_test_graph()
    coarsener = HeteroCoarsener(None,original_graph, 0.5, num_nearest_per_etype=3, num_nearest_neighbors=3,pairs_per_level=30)
    coarsener.init_step()
    for i in range(3):
        print("--------- step: " , i , "---------" )
        coarsener.iteration_step()