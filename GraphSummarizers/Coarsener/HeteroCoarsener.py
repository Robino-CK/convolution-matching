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
from Datasets.NodeClassification.AIFB import AIFB

from Datasets.NodeClassification.TestHetero import TestHeteroSmall, TestHeteroBig
import dgl
import torch
from scipy.sparse import coo_matrix
import scipy
from copy import deepcopy
CHECKPOINTS = [0.7, 0.5, 0.3, 0.1, 0.05, 0.01, 0.001]

class HeteroCoarsener(GraphSummarizer):
    def __init__(self, dataset: Dataset, original_graph: dgl.DGLGraph, r: float, pairs_per_level: int = 10, 
                 num_nearest_neighbors: int = 10
                 ):
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
        self.num_nearest_neighbors = num_nearest_neighbors
        self.pairs_per_level = pairs_per_level
        self.coarsened_graph = original_graph.clone()
        self.merge_graph = None
        self.node_types = dict()
        self.candidate_pairs = None
        self.homo_hetero_map = dict()
        self.type_homo_mapping = dict()
        self.H_originals_stacked = dict()
        self.H_originals = dict()
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
    
        
    def _create_h_spatial_rgcn(self, g):
        start_time = time.time()
        print("start create H")
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
        print("created H", time.time() - start_time )  
    
    
    
    def _init_merge_graph(self, costs):
        print("start init merge graph")
        start_time = time.time()
        self.merge_graphs =dict()
        for ntype in self.coarsened_graph.ntypes:
            num_nodes = self.coarsened_graph.number_of_nodes(ntype=ntype) 
            num_edges = num_nodes * self.num_nearest_neighbors 
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
                    
            if not edge_cnt == num_edges:
                print("WARNING", ntype, "edge count", edge_cnt, "num edges", num_edges)
            edge_tensor = edge_tensor[:, :edge_cnt]
            edge_weight_tensor = edge_weight_tensor[:edge_cnt]
            self.merge_graphs[ntype].add_edges(edge_tensor[0], edge_tensor[1])
            self.merge_graphs[ntype].edata["edge_weight"] = torch.tensor(edge_weight_tensor)
            print("created merge graph for type", ntype)
        print("stop init merge graph", time.time() - start_time)

    def _init_costs_rgcn(self):
        print("start init costs")
        start_time = time.time()
        init_costs = dict()
        
        self.nearest_neighbors_keep_rate = 0.1
        
        for src_type, etype, dst_type in self.coarsened_graph.canonical_etypes:
            if not src_type in init_costs:
                init_costs[src_type] = dict()    
            if "feat" in self.coarsened_graph.nodes[dst_type].data:
                H = torch.zeros(( len(self.coarsened_graph.nodes(src_type)),  self.coarsened_graph.nodes[dst_type].data["feat"].shape[1]))
            else:
                H = torch.zeros(( len(self.coarsened_graph.nodes(src_type)), 1))
            for node in self.coarsened_graph.nodes(src_type):
                
                H[node.item(),:] = self.H_originals[etype][node.item()]
            if H.shape[1] > 3:
                pca = PCA(n_components=3)
                H = pca.fit_transform(
                (H - H.mean(dim=0)) / (H.std(dim=0) + 0.0001)
                )   
            kd_tree = scipy.spatial.KDTree(H)
            distances, nearest_neighbors =  kd_tree.query(H, k=self.num_nearest_neighbors, p=1, eps=0.01, workers=1)
            init_costs[etype] = (distances, nearest_neighbors)
            
        print("stop init costs", time.time() - start_time)
        return init_costs

    def _find_lowest_cost_edges(self):
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
        return topk_non_overlapping_per_type

    def _candidaes_over_all(self, init_costs):
        start_time = time.time()
        total_costs = dict()
        for src_type, etype, dst_type in self.coarsened_graph.canonical_etypes:
            if not src_type in total_costs:
                total_costs[src_type] = list()
                for node in self.coarsened_graph.nodes(src_type):
                    
                    nodes = list(init_costs[etype][1][node.item()])
                    for merge_node in (nodes):
                        if merge_node == node:
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
                mask_node1 =  torch.where(edges_original[0] == mapping[node1], True, False)
                mask_node2 =  torch.where(edges_original[0] == mapping[node2], True, False)
                mask = torch.logical_or(mask_node1, mask_node2)
                edges_dst = torch.unique(edges_original[1][mask])
                edges_src = torch.full(edges_dst.shape,new_node_id )
                g.add_edges(edges_src, edges_dst, etype=(src_type, etype,dst_type))
            
            
            if "feat" in g.nodes[node_type].data:
                old_feats = g.nodes[node_type].data["feat"] 
                g.nodes[node_type].data["feat"][new_node_id] = (old_feats[mapping[node1]] + old_feats[mapping[node2]]) / 2
            if "label" in g.nodes[node_type].data:
                g.nodes[node_type].data["label"][new_node_id] = g.nodes[node_type].data["label"][mapping[node1]]
                
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
            
                
        print("stop merging nodes", time.time()- start_time)       
        return g, mapping     
   


    
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
        
        mapping = torch.arange(0, g.num_nodes() )
        nodes_need_edge_recalc = set()
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
            mapping[node1] = new_node_id  
            mapping[node2] = new_node_id
                
            if node1 > node2:
                mapping = torch.where(mapping >pre_node1, mapping -1, mapping)
                mapping = torch.where(mapping > pre_node2, mapping -1, mapping)
            else:
                mapping = torch.where(mapping > pre_node2, mapping -1, mapping)
                mapping = torch.where(mapping > pre_node1, mapping -1, mapping)
            nodes_need_edge_recalc.add(node1)
            nodes_need_edge_recalc.add(node2)
            
            g.remove_nodes([pre_node1, pre_node2]) 
                
        # TODO maybe neighbors of neighbors of neighbrs
        for old_node in nodes_need_edge_recalc:
            node1 = mapping[old_node]
            for node2 in g.successors(node1):
                costs = 0
                for src_type,etype,dst_type in self.coarsened_graph.canonical_etypes:
                    if src_type != ntype:
                        continue
                    H_type_orig = self.H_originals_stacked[etype]
                    
                    h_uv = self._create_h_spatial_via_cache_for_node(self.coarsened_graph,  node1.item(), (src_type, etype, dst_type), node2.item())
                    costs += torch.norm(h_uv - H_type_orig[node1], 2) + torch.norm(h_uv - H_type_orig[node2], 2)
                edge_id = g.edge_ids(node1, node2)
                g.edata["edge_weight"][edge_id] = costs
        
              
        print("update nerge graph", time.time()- start_time)       
        return g, mapping     
   
    
            
            
    def _costs_of_merges(self, merge_list):
        start_time = time.time()
        
        costs_dict = dict()
        for node_type, merge_candidates in merge_list.items():
            costs_dict[node_type] = dict()
            for src_type,etype,dst_type in self.coarsened_graph.canonical_etypes:
                if src_type != node_type:
                        continue
                
                H_type_orig = self.H_originals_stacked[etype]
                
                for node1, node2 in tqdm(merge_candidates, "calculate H_coarsen"):
                    
                    
                    h_uv = self._create_h_spatial_via_cache_for_node(self.coarsened_graph, node1, (src_type, etype, dst_type), node2)
                    costs = torch.norm(h_uv - H_type_orig[node1], 2) + torch.norm(h_uv - H_type_orig[node2], 2)
                    if (node1, node2) in costs_dict[node_type].keys():
                        costs_dict[node_type][(node1, node2)] += costs
                    else:
                        costs_dict[node_type][(node1, node2)] = costs
        print("costs of merges", time.time() - start_time)
        return costs_dict
       
    
    def _get_master_mapping(self, mappings, ntype):
        master_mapping = dict()
        
        for node in self.original_graph.nodes(ntype):
            node_id = node.item()
            for mapping in mappings:
                node_id = mapping[node_id].item()
            master_mapping[node.item()] = node_id
        return master_mapping
    
    
    def init_step(self):
        self.mappings = dict()
        for ntype in self.coarsened_graph.ntypes:
            self.mappings[ntype] = list()
        
        self._create_h_spatial_rgcn(self.original_graph)
        self.merge_edges = self._costs_of_merges(self._candidaes_over_all(self._init_costs_rgcn()))
        self._init_merge_graph(self.merge_edges)
        self.candidates = self._find_lowest_cost_edges()
    
    
    def iteration_step(self):
        for ntype, merge_list in self.candidates.items():
            self.coarsened_graph, mapping = self._merge_nodes(self.coarsened_graph, ntype, merge_list)
            self.mappings[ntype].append(mapping)
                
            self.merge_graphs[ntype],_ = self._update_merge_graph(self.merge_graphs[ntype], merge_list, ntype)
                
        self.candidates = self._find_lowest_cost_edges()


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

# coarsener = HeteroCoarsener(None,original_graph, 0.5, num_nearest_neighbors=2)
# merge_graph, mapping = coarsener.summarize()