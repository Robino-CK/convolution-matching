import torch
import dgl
from torch_geometric.data import Data
from torch_geometric.data import HeteroData
def from_dgl_hetero_manual(g: dgl.DGLHeteroGraph) -> HeteroData:
    """
    Convert a DGL heterogeneous graph into a PyG HeteroData object.

    Args:
        g (dgl.DGLHeteroGraph): Input DGL heterogeneous graph with node/edge features stored in ndata/edata.

    Returns:
        HeteroData: A PyG HeteroData object with the same node and edge features.
    """
    data = HeteroData()

    # Convert node features for each node type
    for ntype in g.ntypes:
        # Collect all feature fields for this node type
        ndata = g.nodes[ntype].data
        # If there is a 'feat' entry, map it to .x in PyG
        if 'feat' in ndata:
            data[ntype].feat = ndata['feat']
        # Map any other node data fields directly
        for key, value in ndata.items():
            if key != 'feat':
                data[ntype][key] = value

    # Convert edges for each canonical edge type
    for c_etype in g.canonical_etypes:
        src_type, etype, dst_type = c_etype
        # Get edge index (source, target)
        src_nodes, dst_nodes = g.edges(etype=c_etype)
        edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
        # Assign edge_index
        data[c_etype].edge_index = edge_index

        # Collect edge features
        edata = g.edges[c_etype].data
        # If there is a 'feat' entry, map it to .edge_attr
        if 'feat' in edata:
            data[c_etype].edge_attr = edata['feat']
        # Map any other edge data fields directly
        for key, value in edata.items():
            if key != 'feat':
                data[c_etype][key] = value

    return data

def dgl_to_pyg_input(g):
    # Convert DGL heterograph to PyG's HeteroData
    pyg_data = from_dgl_hetero_manual  (g)

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
