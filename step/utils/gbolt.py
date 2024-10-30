import random

import dgl
import numpy as np
import torch


class MultiGraphsAllNodesSampler(dgl.dataloading.SAINTSampler):
    """
    A custom sampler that samples all nodes from sampled graphs.

    Attributes:
        n_graphs: int   Number of graphs to sample from
    """

    def __init__(self, mode, budget, n_graphs=2, ratio=1.):
        super(MultiGraphsAllNodesSampler, self).__init__(mode, budget)
        self.n_graphs = n_graphs
        self.split_ratio = ratio

    def node_sampler(self, g):
        """Node ID sampler for random node sampler"""
        # Assuming g is batched using dgl.batch()
        graphs = dgl.unbatch(g)  # split the batched graph into a list

        sampled_node_ids = []  # create an empty list to store sampled node ids

        # Store the offsets of each unbatched graph
        offsets = [0]
        for graph in graphs:
            offsets.append(offsets[-1] + graph.number_of_nodes())

        # Shuffle the graph indices
        indices = list(range(len(graphs)))
        random.shuffle(indices)

        # Get all nodes from two randomly selected graphs
        for i in indices[: self.n_graphs]:
            graph = graphs[i]
            offset = offsets[i]

            # Include all nodes from the current graph
            if self.split_ratio < 1.0:
                n_nodes = int(graph.number_of_nodes() * self.split_ratio)
                sampled_nodes = self._single_graph_node_sampler(graph, n_nodes)
            elif self.split_ratio == 1.0:
                sampled_nodes = torch.arange(
                    graph.number_of_nodes(), dtype=torch.long
                )
            else:
                assert isinstance(self.split_ratio, int), "Get integer split ratio, deemed as number of nodes"
                sampled_nodes = self._single_graph_node_sampler(graph, self.split_ratio)
            sampled_node_ids.append(
                sampled_nodes + offset
            )  # Adjust node IDs based on offset

        # Concatenate all the node IDs into one tensor
        sampled_node_ids_tensor = torch.cat(sampled_node_ids)

        return sampled_node_ids_tensor

    def _single_graph_node_sampler(self, g, n_nodes):
        prob = g.out_degrees().float().clamp(min=1)
        return (
            torch.multinomial(prob, num_samples=n_nodes, replacement=True)
            .unique()
            .type(g.idtype)
        )
