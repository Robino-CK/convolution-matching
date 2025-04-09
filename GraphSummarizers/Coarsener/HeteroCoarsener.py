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
        
    def _create_h_spectral_rgcn(self, g):
        # TODO: no normalization :// 
        H = dict()
        start_time = time.time()
        for src_type, etype, dst_type in g.canonical_etypes:
            
            adj = g.adj_external(etype=etype)
            d_src = torch.pow(torch.diag(adj.sum(axis=0).to_dense()),-0.5) 
            d_inv_sqrt_src = torch.where(d_src == float("inf"), d_src, torch.tensor(0.0))
            d_dst = torch.pow(torch.diag(adj.sum(axis=1).to_dense()),-0.5)
            d_inv_sqrt_dst = torch.where(d_dst == float("inf"), d_dst, torch.tensor(0.0))
            
            H[etype] =  d_inv_sqrt_dst@ adj @ d_inv_sqrt_src
            H[etype] = H[etype].to_dense()

        print("created H", time.time() - start_time)
        return H
            
    def _create_h_spatial_rgcn_for_node(self, g, node, node_type):
        h = dict()
        
            
        for src_type, etype, dst_type in g.canonical_etypes:
            is_features = "feat" in g.nodes[dst_type].data
            if node_type != src_type:
                continue
            degree = torch.tensor(self.node_degrees[etype]["out"]) 
            degree_inv_src = 1/torch.sqrt(degree)
            degree = torch.tensor(self.node_degrees[etype]["in"]) 
            degree_inv_dest = 1/torch.sqrt(degree)
            neighbors = g.successors(node, etype=(src_type, etype, dst_type))
            if is_features:
                h[etype] = torch.zeros(g.nodes[dst_type].data["feat"].shape[1])
            else:
                h[etype] = torch.zeros(1)
            for neigh in neighbors:
                if is_features:
                    h[etype] += degree_inv_src[node] * degree_inv_dest[neigh] * g.nodes[dst_type].data["feat"][neigh]
                else:
                    h[etype] += degree_inv_src[node] * degree_inv_dest[neigh] 
        return h



    def _create_h_spatial_rgcn(self, g):
        start_time = time.time()
        print("start create H")
        H = dict()
        device = "cpu"
        
        for src_type, etype, dst_type in g.canonical_etypes:
            is_features = "feat" in g.nodes[dst_type].data
            degree = torch.tensor(self.node_degrees[etype]["out"]).to(device) + 1 
            degree_inv_src = 1/torch.sqrt(degree) 
            degree_inv_src = degree_inv_src.to(device)
            degree_inv_src = torch.where(degree_inv_src == float("inf"),  torch.tensor(0.0),degree_inv_src) # why does this not work
           
            degree = torch.tensor(self.node_degrees[etype]["in"]).to(device) + 1 
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
                
        print("created H", time.time() - start_time )  
        return H
    
    
    
    def _get_rgcn_edges(self, H_normal):
        print("start init costs")
        start_time = time.time()
        init_costs = dict()
        
        self.nearest_neighbors_keep_rate = 0.1
        self.top_k_nn = 3
        self.num_nearest_neighbors = 30
        
        for src_type, etype, dst_type in self.coarsened_graph.canonical_etypes:
            if not src_type in init_costs:
                init_costs[src_type] = dict()    
            if "feat" in self.coarsened_graph.nodes[dst_type].data:
                H = torch.zeros(( len(self.coarsened_graph.nodes(src_type)),  self.coarsened_graph.nodes[dst_type].data["feat"].shape[1]))
            else:
                H = torch.zeros(( len(self.coarsened_graph.nodes(src_type)), 1))
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
        print("stop init costs", time.time() - start_time)
        return init_costs
    
    def _sum_edge_costs(self, init_costs):
        total_costs = dict()
        for src_type, etype, dst_type in self.coarsened_graph.canonical_etypes:
            if not src_type in total_costs:
                total_costs[src_type] = dict()
                for node in self.coarsened_graph.nodes(src_type):
                    if not node.item() in total_costs[src_type]:
                        total_costs[src_type][node.item()] = dict()
                    
                    costs = list(init_costs[etype][0][node.item()])
                    nodes = list(init_costs[etype][1][node.item()])
                    for cost, merge_node in zip(costs, nodes):
                        if merge_node == node:
                            continue
                        if not merge_node in total_costs[src_type][node.item()]:
                            total_costs[src_type][node.item()][merge_node] = cost
                        else:
                            total_costs[src_type][node.item()][merge_node] += cost
        return total_costs
    
    def _find_lowest_lowest_k_cost_edges(self, total_costs, k =200):
        lowest_nodes_per_type = dict()
        for ntype in self.coarsened_graph.ntypes:
            if ntype not in total_costs:
                continue
            lowest_costs = [float("inf")]*k
            lowest_nodes = [None]*k
                
            for node, candidates in total_costs[ntype].items():
                
                for merge_node, cost in candidates.items():
                    if (node, merge_node) in lowest_nodes:
                       continue
             
                    if cost < lowest_costs[-1]:
                        lowest_costs[-1] = cost
                        lowest_nodes[-1] = (merge_node, node)
                        lowest_costs, lowest_nodes = zip(*sorted(zip(lowest_costs, lowest_nodes)))
                        lowest_costs, lowest_nodes = list(lowest_costs), list(lowest_nodes)
            
            # only allow merging of nodes that are not already merged
            all_merged = set()
            for node_pair in lowest_nodes:
                if node_pair == None:
                    continue
                node1, node2 = node_pair
                if node1 in all_merged or node2 in all_merged:
                    continue
                all_merged.add(node1)
                all_merged.add(node2)
            lowest_without_dups = list()
            for node_pair in lowest_nodes:
                if node_pair == None:
                    continue
                node1, node2 = node_pair
                if node1 in all_merged and node2 in all_merged:
                    lowest_without_dups.append((node1, node2))
                    all_merged.remove(node1)
                    all_merged.remove(node2)
            lowest_nodes_per_type[ntype] = lowest_without_dups
        return lowest_nodes_per_type                        
                
              
    def _get_intersection(self, init_costs):
        print("start lowest cost edges")
        start_time = time.time()
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
        print("stop lowest cost edges", time.time() - start_time)
        return closest_over_all_etypes

    
    def _select_candidates(self, closest_over_all_etypes):
        print("start select candidates")
        start_time = time.time()
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
        print("stop select canidates", time.time() - start_time)                
        return merge_list

    
    def _merge_nodes(self, g, node_type, node_pairs, mappings):
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
            
                
    #   print("stop merging nodes", time.time()- start_time)       
        return g, mapping     
   
    
            
    def _costs_of_merges(self, merge_list, H):
        H_original = H#self._create_h_spatial_rgcn(self.coarsened_graph)
        costs_dict = dict()
        for node_type, merge_candidates in merge_list.items():
            costs_dict[node_type] = dict()
            if node_type != "author":
                continue
            for node1, node2 in tqdm(merge_candidates, "calculate H_coarsen"):
                
                merged_graph, mapping_authors = self._merge_nodes(self.coarsened_graph, node_type, [(node1, node2)])
              #  P = self._get_partitioning(node2, node1, self.coarsened_graph.num_nodes(ntype=node_type))
                h_u = self._create_h_spatial_rgcn_for_node(merged_graph, mapping_authors[node1], node_type)
                h_v = self._create_h_spatial_rgcn_for_node(merged_graph, mapping_authors[node2], node_type)
                
                costs = 0
                for src_type,etype,dst_type in self.coarsened_graph.canonical_etypes:
                    
                    if src_type != node_type:
                        continue
                    H_type_orig = torch.stack(list(H_original[etype].values()))
                    
                    costs = torch.norm(h_u[etype] - H_type_orig[node1], 2) + torch.norm(h_v[etype] - H_type_orig[node2], 2)
                    
                  #  neighbors1 = self.coarsened_graph.successors(node1, etype=etype)
                 #   neighbors2 = self.coarsened_graph.successors(node2, etype=etype)
                 #   neighbors = set(torch.cat((neighbors1, neighbors2)).tolist())
                    
        #            for neighbor in neighbors:
       #                 h_i = self._create_h_spatial_rgcn_for_node(self.coarsened_graph, neighbor, dst_type)
                        
                       # costs += torch.norm(h_i[etype] - H_something[neighbor], 2)
                costs_dict[node_type][(node1, node2)] = costs
        return costs_dict
       
    def _get_partitioning(self, u,v , number_of_nodes):
        P = torch.eye(number_of_nodes)
        P = np.delete(P, number_of_nodes - 1, axis=1)
        P[u,v] = 1
        if not u == number_of_nodes - 1:
            P[u,u] = 0
            P[-1, u]  = 1  
        return P

    def _get_candidates(self, original_graph):
        
        pass
    
    def summarize(self, original_graph):
        H = self._create_h_spatial_rgcn(original_graph)
        candidates = self._select_candidates(self._get_intersection(self._get_rgcn_edges(H)))
        #return candidates
        costs = self._costs_of_merges(candidates, H)["author"]
        sort_by_value = dict(sorted(costs.items(), key=lambda item: item[1]))
        to_merge = []
        i = 0
        for key, value in sort_by_value.items():
            if i > 500:
                continue
            to_merge.append(key)
            i += 1
        merged_graph, mapping_authors = self._merge_nodes(original_graph, "author", to_merge)


        return merged_graph, mapping_authors
    
        
    def summarize2(self, k=5):
        
        for i in range(k):
            H = self._create_h_spatial_rgcn(self.coarsened_graph)
            candidates = self._find_lowest_lowest_k_cost_edges(self._sum_edge_costs(self._get_rgcn_edges(H)))
            for ntype in self.coarsened_graph.ntypes:
                if self.coarsened_graph.num_nodes(ntype) * 2 < self.original_graph.num_nodes(ntype):
                    continue
                self.coarsened_graph, mapping_authors = self._merge_nodes(self.coarsened_graph, ntype, candidates[ntype], None)

        return self.coarsened_graph, mapping_authors
    
    

# dataset = DBLP() 
# original_graph = dataset.load_graph()

# coarsener = HeteroCoarsener(None,g, 0.5)


# coarsener.summarize2()

#H = coarsener._create_h_spatial_rgcn(original_graph)
#candidates = coarsener._find_lowest_lowest_k_cost_edges(coarsener._sum_edge_costs(coarsener._get_rgcn_edges(H)))
#candidates = coarsener._select_candidates(coarsener._get_intersection(coarsener._get_rgcn_edges(H)))
#merged_graph, mapping_authors = coarsener._merge_nodes(original_graph, "author", candidates["author"])
# #merged_graph, mapping_paper = coarsener._merge_nodes(merged_graph, "paper", candidates["paper"])
# print(merged_graph)
# print(original_graph)
#coarsener._costs_of_merges(candidates)

#coarsener._merge()
# aifb = AIFB()
# new_g = aifb.load_graph()   
    
# new_g.nodes("Forschungsgebiete")

# coarsener = HeteroCoarsener(None, new_g, 0.5)
# H = coarsener._create_h_spatial_rgcn(new_g)
# candidates = coarsener._select_candidates(coarsener._find_lowest_cost_edges(coarsener._get_rgcn_edges(H)))
# merged_graph, mapping_authors = coarsener._merge_nodes(new_g, "Forschungsgebiete", candidates["Forschungsgebiete"])

#
    
    

#dataset = DBLP() 
#original_graph = dataset.load_graph()


#test = TestHeteroSmall().load_graph()

# test = TestHeteroBig().load_graph()

#coarsener = HeteroCoarsener(dataset, original_graph, 0.5)
#
#coarsener._create_h_spectral_rgcn(original_graph)
# coarsener.summarize(original_graph)
