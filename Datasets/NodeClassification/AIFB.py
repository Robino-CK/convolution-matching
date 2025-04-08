import dgl
import dgl
import torch


class AIFB():
    def __init__(self):
        pass

    def load_graph(self):
        # Load the dataset
        dataset = dgl.data.rdf.AIFBDataset()
        g = dataset[0]
        category = dataset.predict_category

        # Suppose g is your existing heterograph.
        # For example:
        # g = dgl.heterograph({
        #     ('user', 'rates', 'item'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])),
        #     ('item', 'belongs', 'category'): (torch.tensor([1, 2]), torch.tensor([0, 1]))
        # })

        # Dictionary to hold connectivity information for the new graph.
        new_graph_data = {}

        # Loop over each canonical edge type in the original graph.
        for canonical in g.canonical_etypes:
            src_type, rel_type, dst_type = canonical
            src_ids, dst_ids = g.edges(etype=canonical)
            
            # Create the new relation name: src_rel_dst.
            new_rel = f"{src_type}_{rel_type}_{dst_type}"
            new_key = (src_type, new_rel, dst_type)
            
            # Add the connectivity information to the dictionary.
            new_graph_data[new_key] = (src_ids, dst_ids)

        # Construct the new heterograph using the updated keys.
        new_g = dgl.heterograph(new_graph_data)

        # Copy edge features from the original graph to the new graph.
        for canonical in g.canonical_etypes:
            src_type, rel_type, dst_type = canonical
            new_rel = f"{src_type}_{rel_type}_{dst_type}"
            new_key = (src_type, new_rel, dst_type)
            
            # Transfer each feature for this edge type.
            for feat_name, feat_data in g.edges[canonical].data.items():
                new_g.edges[new_key].data[feat_name] = feat_data

        # Copy node features from the original graph to the new graph.
        for ntype in g.ntypes:
            for feat_name, feat_data in g.nodes[ntype].data.items():
                new_g.nodes[ntype].data[feat_name] = feat_data

        # Now, new_g has all edge types renamed with the pattern: srcType_oldRel_dstType.
        return new_g