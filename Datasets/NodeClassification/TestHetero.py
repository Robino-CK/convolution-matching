import dgl
import torch
import random
class TestHeteroSmall():
    def __init__(self):
        pass
    def load_graph(self):
    # Define the number of nodes for each type

        num_authors = 3
        num_papers = 3

        # Define edges: (source_nodes, destination_nodes)
        # Author 0 writes Paper 0, Author 1 writes Paper 1 and 2, Author 2 writes Paper 0
        author_ids = {
            'writes': torch.tensor([0, 1, 1, 2]),  # authors
            'written_by': torch.tensor([0, 1, 2, 0])  # papers
        }
        paper_ids = {
            'writes': torch.tensor([0, 1, 2, 0]),  # papers
            'written_by': torch.tensor([0, 1, 1, 2])  # authors
        }

        # Create the heterograph
        hetero_graph = dgl.heterograph({
            ('author', 'writes', 'paper'): (author_ids['writes'], paper_ids['writes']),
            ('paper', 'written_by', 'author'): (paper_ids['written_by'], author_ids['written_by'])
        })
        return hetero_graph
    
    
class TestHeteroBig():
    def __init__(self):
        pass
    def load_graph(self):
        # Node type sizes
        num_authors = 40
        num_papers = 40
        num_conferences = 20

        # Total nodes per type
        num_nodes = {
            'author': num_authors,
            'paper': num_papers,
            'conference': num_conferences
        }

        # Random edges: authors write papers
        num_edges_write = 100
        author_write = torch.randint(0, num_authors, (num_edges_write,))
        paper_written = torch.randint(0, num_papers, (num_edges_write,))

        # Random edges: papers published in conferences
        num_edges_publish = 80
        paper_publish = torch.randint(0, num_papers, (num_edges_publish,))
        conf_publish = torch.randint(0, num_conferences, (num_edges_publish,))

        # Random edges: authors collaborate with other authors
        num_edges_collab = 60
        author1 = torch.randint(0, num_authors, (num_edges_collab,))
        author2 = torch.randint(0, num_authors, (num_edges_collab,))

        # Construct heterograph
        graph = dgl.heterograph({
            ('author', 'writes', 'paper'): (author_write, paper_written),
            ('paper', 'published_in', 'conference'): (paper_publish, conf_publish),
            ('author', 'collaborates', 'author'): (author1, author2),
        }, num_nodes_dict=num_nodes)

        # Add node features (10-dimensional)
        feat_dim = 10
        graph.nodes['author'].data['feat'] = torch.randn(num_authors, feat_dim)
        graph.nodes['paper'].data['feat'] = torch.randn(num_papers, feat_dim)
        graph.nodes['conference'].data['feat'] = torch.randn(num_conferences, feat_dim)

        # Add node labels (e.g., for classification)
        num_classes = 3
        graph.nodes['author'].data['label'] = torch.randint(0, num_classes, (num_authors,))
        graph.nodes['paper'].data['label'] = torch.randint(0, num_classes, (num_papers,))
        graph.nodes['conference'].data['label'] = torch.randint(0, num_classes, (num_conferences,))
        return graph