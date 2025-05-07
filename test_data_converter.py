import torch

import torch
from torch_geometric.data import Data

def from_dgl(g):
    # edges
    u, v = g.edges()
    edge_index = torch.stack([u, v], dim=0)
    # node features (if any)
    x = g.ndata.get('x', None)
    # edge features (if any)
    edge_attr = g.edata.get('edge_attr', None)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def dgl_to_pyg_input(g):
    # Convert DGL heterograph to PyG's HeteroData
    pyg_data = from_dgl(g)

    x_dict = {}
    for ntype in g.ntypes:
        if 'feat' in g.nodes[ntype].data:
            x_dict[ntype] = g.nodes[ntype].data['feat']

    edge_index_dict = {}
    for canonical_etype in g.canonical_etypes:
        src_type, rel_type, dst_type = canonical_etype
        src, dst = g.edges(etype=canonical_etype)
        edge_index = torch.stack([src, dst], dim=0)
        edge_index_dict[(src_type, rel_type, dst_type)] = edge_index
    pyg_data.edge_index_dict = edge_index_dict
    return pyg_data,x_dict, edge_index_dict, g.ntypes, g.canonical_etypes
