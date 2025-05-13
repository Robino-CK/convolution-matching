import numpy as np
import torch
import math
import dgl
class TestHomo(): 
    def __init__(self):
        self.g = self.create_test_graph()
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
        
        
    def run_test(self, coarsener):
       
        self.coarsener = coarsener
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
        
        
class TestHetero():
    def __init__(self):
        self.g  = self.create_test_graph()
        self.s = dict()
        self.s["writes"] = dict()
        self.s["writes"][0] = (1 / np.sqrt(3)  * torch.tensor([0,1])) 
        self.s["writes"][1] = (1 / np.sqrt(3) * torch.tensor([0,1] )+  1 / np.sqrt(3) * torch.tensor([1,2]))
        self.s["writes"][2] = (1 / np.sqrt(3) * torch.tensor([1,2]))

        self.s["cites"] = dict()
        self.s["cites"][0] = (1  / np.sqrt(2) * torch.tensor([1,2]) ) 
        self.s["cites"][1] = torch.tensor([0,0])
        
        self.h = dict()
        self.h["writes"] = dict()
        self.h["writes"][0] = (1 / np.sqrt(2)  * self.s["writes"][0]) 
        self.h["writes"][1] = (1 / np.sqrt(3) *  self.s["writes"][1])
        self.h["writes"][2] = (1 / np.sqrt(2) *  self.s["writes"][2])
        
        
        self.h["cites"] = dict()
        self.h["cites"][0] = (1  / np.sqrt(2) * self.s["cites"][0] ) 
        self.h["cites"][1] = torch.tensor([0,0])
        
        self.nearest_neighbors = dict()
        self.nearest_neighbors["author"] = {0: {1,2}, 1:{0,2}, 2:{1}}
        self.nearest_neighbors["paper"] = {0: {1}, 1:{0}}
     
     
    def create_test_graph(self):
        g = dgl.heterograph({
            ('author', 'writes', 'paper'): ([0, 1, 1, 2], [0, 0, 1, 1]),
            ('paper', 'cites', 'paper'): ([0], [1])
            })
        g.nodes['author'].data['feat'] = torch.tensor([[1.0],[2.0],[3.0]])
        g.nodes['paper'].data['feat'] = torch.tensor([[0.0, 1.0], [1.0, 2.0]])
        

        return g
    
    
    def check_H(self):
        for k, v in self.s["writes"].items():
            assert math.isclose(self.coarsener.coarsened_graph.nodes["author"].data[f'swrites'][k][0].item(), v[0], rel_tol=1e-6), f"error in creating H for {k}"
            assert math.isclose(self.coarsener.coarsened_graph.nodes["author"].data[f'swrites'][k][1].item(), v[1], rel_tol=1e-6), f"error in creating H for {k}"
            
        for k, v in self.h["writes"].items():
            assert math.isclose(self.coarsener.coarsened_graph.nodes["author"].data[f'hwrites'][k][0].item(), v[0], rel_tol=1e-6), f"error in creating H for {k}"
            assert math.isclose(self.coarsener.coarsened_graph.nodes["author"].data[f'hwrites'][k][1].item(), v[1], rel_tol=1e-6), f"error in creating H for {k}"
            
        for k, v in self.s["cites"].items():
            assert math.isclose(self.coarsener.coarsened_graph.nodes["paper"].data[f'scites'][k][0].item(), v[0], rel_tol=1e-6), f"error in creating H for {k}"
            assert math.isclose(self.coarsener.coarsened_graph.nodes["paper"].data[f'scites'][k][1].item(), v[1], rel_tol=1e-6), f"error in creating H for {k}"
            
        for k, v in self.h["cites"].items():
            assert math.isclose(self.coarsener.coarsened_graph.nodes["paper"].data[f'hcites'][k][0].item(), v[0], rel_tol=1e-6), f"error in creating H for {k}"
            assert math.isclose(self.coarsener.coarsened_graph.nodes["paper"].data[f'hcites'][k][1].item(), v[1], rel_tol=1e-6), f"error in creating H for {k}"
    
    def check_init_feat_costs(self):
        costs = self.coarsener.init_costs_dict_features["author"]["costs"]
        index = self.coarsener.init_costs_dict_features["author"]["index"]
        self.correct_feat_costs = dict()
        self.correct_feat_costs["author"] = {(0,1): 1, (1,0): 1 ,  (0,2): 2, (1,2): 1, (2,1):1}
        for i in range(costs.shape[0]):
            node1 = index[0][i].item()
            node2 = index[1][i].item()
            cost = costs[i].item()
            if (node1, node2) in self.correct_feat_costs["author"]:
                assert math.isclose(cost, self.correct_feat_costs["author"][(node1, node2)], rel_tol=1e-6), f"error in init feat costs"
            else:
                assert math.isclose(cost, 0.0, rel_tol=1e-6), f"error in init feat costs {cost} != 0"
            assert len(self.correct_feat_costs["author"]) == costs.shape[0], f"error in init feat costs"
            
        costs = self.coarsener.init_costs_dict_features["paper"]["costs"]
        index = self.coarsener.init_costs_dict_features["paper"]["index"]
        self.correct_feat_costs["paper"] = {(0,1):2, (1,0): 2 }
        for i in range(costs.shape[0]):
            node1 = index[0][i].item()
            node2 = index[1][i].item()
            cost = costs[i].item()
            if (node1, node2) in self.correct_feat_costs["paper"]:
                assert math.isclose(cost, self.correct_feat_costs["paper"][(node1, node2)], rel_tol=1e-6), f"error in init feat costs "
            else:
                assert math.isclose(cost, 0.0, rel_tol=1e-6), f"error in init feat costs {cost} != 0"
            assert len(self.correct_feat_costs["paper"]) == costs.shape[0], f"error in init feat costs "
    
    
    
    
    def check_init_H_costs(self):
        
        self.correct_H_costs = { }
        self.correct_H_costs["author"] = dict()
        for k, list_values in self.nearest_neighbors["author"].items():
            for v in list_values:
                
                huv = (self.s["writes"][k] + self.s["writes"][v]) / np.sqrt(self.g.out_degrees(k, "writes") + self.g.out_degrees(v, "writes") + 2)
                self.correct_H_costs["author"][k,v] =  torch.norm(torch.tensor(huv - self.h["writes"][k]),  p=1).item() + torch.norm(torch.tensor(huv -self.h["writes"][v]),  p=1).item()
        costs = self.coarsener.init_costs_dict_etype["author"]["writes"]["costs"]
        index = self.coarsener.init_costs_dict_etype["author"]["writes"]["index"]
        for i in range(costs.shape[0]):
            node1 = index[0][i].item()
            node2 = index[1][i].item()
            cost = costs[i].item()
            if (node1, node2) in self.correct_H_costs["author"]:
                assert math.isclose(cost, self.correct_H_costs["author"][(node1, node2)], rel_tol=1e-2), f"error in init feat costs"
            else:
                assert math.isclose(cost, 0.0, rel_tol=1e-6), f"error in init feat costs {cost} != 0"
        assert len(self.correct_H_costs["author"]) == costs.shape[0], f"error in init feat costs"
        
        self.correct_H_costs["paper"] = { }
        for k, list_values in self.nearest_neighbors["paper"].items():
            for v in list_values:
                
                huv = (self.s["cites"][k] + self.s["cites"][v]) / np.sqrt(self.g.out_degrees(k, "cites") + self.g.out_degrees(v, "cites") + 2)
                self.correct_H_costs["paper"][k,v] =  torch.norm(torch.tensor(huv - self.h["cites"][k]),  p=1).item() + torch.norm(torch.tensor(huv -self.h["cites"][v]),  p=1).item()
        costs = self.coarsener.init_costs_dict_etype["paper"]["cites"]["costs"]
        index = self.coarsener.init_costs_dict_etype["paper"]["cites"]["index"]
        for i in range(costs.shape[0]):
            node1 = index[0][i].item()
            node2 = index[1][i].item()
            cost = costs[i].item()
            if (node1, node2) in self.correct_H_costs["paper"]:
                assert math.isclose(cost, self.correct_H_costs["paper"][(node1, node2)], rel_tol=1e-2), f"error in init feat costs "
            else:
                assert math.isclose(cost, 0.0, rel_tol=1e-6), f"error in init feat costs {cost} != 0"
        assert len(self.correct_H_costs["paper"]) == costs.shape[0], f"error in init feat costs "
        
    
    
    def check_init_total_costs(self):
        costs = self.coarsener.init_costs_dict["author"]
        index = self.coarsener.init_index_dict["author"]
        correct_costs = dict()
        for k, list_values in self.nearest_neighbors["author"].items():
            for v in list_values:
                correct_costs[k,v] = (self.correct_H_costs["author"][k,v] / 0.1412)  + (self.correct_feat_costs["author"][k,v] / 1)
                
        for i in range(costs.shape[0]):
            node1 = index[0][i].item()
            node2 = index[1][i].item()
            cost = costs[i].item()
            if (node1, node2) in self.correct_H_costs["author"]:
                assert math.isclose(cost, correct_costs[(node1, node2)], rel_tol=1e-2), f"error in init feat costs"
            else:
                assert math.isclose(cost, 0.0, rel_tol=1e-6), f"error in init feat costs"
        assert len(self.correct_H_costs["author"]) == costs.shape[0], f"error in init feat costs"
        
        
        costs = self.coarsener.init_costs_dict["paper"]
        index = self.coarsener.init_index_dict["paper"]
        correct_costs = dict()
        for k, list_values in self.nearest_neighbors["paper"].items():
            for v in list_values:
                correct_costs[k,v] = (self.correct_H_costs["paper"][k,v] / 0.0000000000000000000000000000000000001)  + (self.correct_feat_costs["paper"][k,v] / 0.0000000000000000000000000000000001)
        t = 2    
        # for i in range(costs.shape[0]):
        #     node1 = index[0][i].item()
        #     node2 = index[1][i].item()
        #     cost = costs[i].item()
        #     if (node1, node2) in self.correct_H_costs["paper"]:
        #         assert math.isclose(cost, correct_costs[(node1, node2)], rel_tol=1e-2), f"error in init feat costs {cost} != {correct_costs[(node1, node2)]}"
        #     else:
        #         assert math.isclose(cost, 0.0, rel_tol=1e-6), f"error in init feat costs {cost} != 0"
        # assert len(self.correct_H_costs["paper"]) == costs.shape[0], f"error in init feat costs {costs.shape[0]} != {len(self.correct_H_costs)}"
    
    def check_first_merge_candidates(self):
        candidates = [(0,1)]
        assert self.coarsener.candidates["author"] == candidates, "error first merge candidates"
        assert self.coarsener.candidates["paper"] == candidates, "error first merge candidates"
    
    
   
        
    def check_first_merge_nodes(self):
        assert all(self.coarsener.coarsened_graph.edges(etype="writes")[0] ==  torch.tensor([0,1])), "error merged edges not correct"
        assert all(self.coarsener.coarsened_graph.edges(etype="writes")[1] ==  torch.tensor([0,0]) ), "error merged edges not correct"
        
        assert all(self.coarsener.coarsened_graph.nodes["author"].data["feat"] == torch.tensor([[3.0], [1.5]])), "error merge features not correct"
        assert all(self.coarsener.coarsened_graph.nodes["paper"].data["feat"][0] == torch.tensor([0.5, 1.5]))
        
        
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
        
        
    def run_test(self, coarsener):
       
        self.coarsener = coarsener
        self.coarsener.init_step()
        self.check_H()
        self.check_init_feat_costs()
        self.check_init_H_costs()
        self.check_init_total_costs()
        self.check_first_merge_candidates()
        self.coarsener.iteration_step()
        self.check_first_merge_nodes()
        #self.check_first_merge_s_and_h()
        #self.check_first_merge_features()
        #self.check_first_merge_updated_merge_edges()
        t = 2

    
        
    
    
