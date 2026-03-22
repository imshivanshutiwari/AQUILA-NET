import torch
from torch_geometric.data import Data


class TopologyUpdater:
    """Updates graph topology based on anomaly scores and temporal dependencies."""

    def update_edge_weights(
        self,
        graph_data: Data,
        anomaly_scores: torch.Tensor,
    ) -> Data:
        """Recompute edge weights from the anomaly scores of incident nodes.

        The weight of an edge (u, v) is:
            w = 1 / (1 + (score_u + score_v) / 2)

        This down-weights edges connecting anomalous nodes.

        Args:
            graph_data: graph with edge_index of shape (2, E).
            anomaly_scores: 1-D tensor of shape (N,) – one score per node.

        Returns:
            Updated Data object with ``edge_weight`` attribute set.
        """
        edge_index = graph_data.edge_index
        src, dst = edge_index[0], edge_index[1]

        scores = anomaly_scores.squeeze(-1)
        edge_weight = 1.0 / (1.0 + (scores[src] + scores[dst]) / 2.0)

        graph_data.edge_weight = edge_weight
        return graph_data

    def remove_critical_nodes(
        self,
        graph_data: Data,
        threshold: float = 0.9,
    ) -> Data:
        """Zero-out features of nodes whose anomaly score exceeds *threshold*.

        The node is considered "offline"; its feature vector is replaced with
        zeros so downstream layers receive no signal from it.

        Args:
            graph_data: must have an ``anomaly_scores`` attribute of shape (N,)
                        or (N, 1), or the scores are looked up from
                        ``graph_data.anomaly_scores`` if present.
            threshold: anomaly score above which a node is marked offline.

        Returns:
            Updated Data object with modified x.
        """
        if not hasattr(graph_data, "anomaly_scores") or graph_data.anomaly_scores is None:
            return graph_data

        scores = graph_data.anomaly_scores.squeeze(-1)
        offline_mask = scores > threshold

        x = graph_data.x.clone()
        x[offline_mask] = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
        graph_data.x = x
        graph_data.offline_mask = offline_mask
        return graph_data

    def add_temporal_edges(
        self,
        graph_data: Data,
        time_lag: int = 1,
    ) -> Data:
        """Add self-loop edges to represent temporal dependencies.

        A self-loop on every node encodes the idea that a node's state at
        time *t* depends on its own state at time *t - time_lag*.

        Args:
            graph_data: existing Data object.
            time_lag: kept as metadata; the self-loops themselves are
                      unweighted (edge weight = 1.0 / time_lag).

        Returns:
            Updated Data object with appended self-loops in edge_index (and
            edge_attr if it was present).
        """
        n_nodes = graph_data.x.shape[0]
        self_loops = torch.arange(n_nodes, device=graph_data.edge_index.device)
        self_loop_index = torch.stack([self_loops, self_loops], dim=0)

        graph_data.edge_index = torch.cat(
            [graph_data.edge_index, self_loop_index], dim=1
        )

        if hasattr(graph_data, "edge_attr") and graph_data.edge_attr is not None:
            n_edge_features = graph_data.edge_attr.shape[1]
            # Self-loop edge attributes: zero-filled except first column = time_lag
            self_loop_attr = torch.zeros(
                n_nodes, n_edge_features,
                dtype=graph_data.edge_attr.dtype,
                device=graph_data.edge_attr.device,
            )
            self_loop_attr[:, 0] = float(time_lag)
            graph_data.edge_attr = torch.cat(
                [graph_data.edge_attr, self_loop_attr], dim=0
            )

        graph_data.time_lag = time_lag
        return graph_data
