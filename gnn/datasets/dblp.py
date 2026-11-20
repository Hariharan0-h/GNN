"""
DBLP Dataset Loader

Loads the PyTorch Geometric DBLP dataset and
constructs an author-author collaboration graph.

Two authors are connected when they share a paper.
"""

import torch
import networkx as nx

from torch_geometric.datasets import DBLP


def load_dblp():

    dataset = DBLP(
        root="data/DBLP"
    )

    hetero_data = dataset[0]

    # --------------------------------------------------
    # Author nodes
    # --------------------------------------------------

    author_features = hetero_data["author"].x

    num_authors = author_features.shape[0]

    # --------------------------------------------------
    # Author-paper relationship
    # --------------------------------------------------

    author_paper_edges = hetero_data[
        "author",
        "to",
        "paper"
    ].edge_index

    # --------------------------------------------------
    # Build author-paper mapping
    # --------------------------------------------------

    paper_authors = {}

    edges = author_paper_edges.cpu().numpy()

    for i in range(edges.shape[1]):

        author_id = int(edges[0, i])
        paper_id = int(edges[1, i])

        if paper_id not in paper_authors:

            paper_authors[paper_id] = []

        paper_authors[paper_id].append(
            author_id
        )

    # --------------------------------------------------
    # Build author-author collaboration graph
    # --------------------------------------------------

    G = nx.Graph()

    G.add_nodes_from(
        range(num_authors)
    )

    for authors in paper_authors.values():

        # Remove duplicate authors
        authors = list(set(authors))

        for i in range(len(authors)):

            for j in range(i + 1, len(authors)):

                u = authors[i]
                v = authors[j]

                if u != v:

                    if G.has_edge(u, v):

                        G[u][v]["weight"] += 1

                    else:

                        G.add_edge(
                            u,
                            v,
                            weight=1
                        )

    print("=" * 60)
    print("DBLP Author Collaboration Graph")
    print("=" * 60)

    print(
        "Authors :",
        G.number_of_nodes()
    )

    print(
        "Edges   :",
        G.number_of_edges()
    )

    print(
        "Papers  :",
        len(paper_authors)
    )

    print("=" * 60)

    return G