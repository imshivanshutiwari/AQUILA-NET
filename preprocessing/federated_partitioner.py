import math
from typing import List

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch_geometric.data import Data


class FederatedPartitioner:
    """Partitions graph data across federated learning clients."""

    def __init__(self):
        self._client_data_sizes: List[int] = []

    # ------------------------------------------------------------------
    # Geography-based partitioning
    # ------------------------------------------------------------------

    def partition_by_geography(self, graph_data: Data, n_clients: int) -> List[Data]:
        """Split graph nodes into geographic partitions using k-means.

        Node features are assumed to have latitude at column 0 and longitude
        at column 1.

        Args:
            graph_data: full graph Data object.
            n_clients: number of client partitions to produce.

        Returns:
            List of *n_clients* Data objects, one per partition.
        """
        x = graph_data.x
        n_nodes = x.shape[0]
        n_clients = min(n_clients, n_nodes)

        coords = x[:, :2].detach().cpu().numpy()  # (lat, lon)

        kmeans = KMeans(n_clusters=n_clients, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)

        edge_index = graph_data.edge_index
        edge_attr = graph_data.edge_attr if hasattr(graph_data, "edge_attr") else None

        partitions: List[Data] = []
        self._client_data_sizes = []

        for client_id in range(n_clients):
            node_mask = torch.tensor(labels == client_id, dtype=torch.bool)
            node_indices = node_mask.nonzero(as_tuple=True)[0]

            # Remap global node indices to local ones
            global_to_local = {int(g): l for l, g in enumerate(node_indices.tolist())}

            # Filter edges where both endpoints belong to this partition
            if edge_index.shape[1] > 0:
                src, dst = edge_index[0], edge_index[1]
                edge_mask = node_mask[src] & node_mask[dst]
                local_src = torch.tensor(
                    [global_to_local[int(s)] for s in src[edge_mask].tolist()],
                    dtype=torch.long,
                )
                local_dst = torch.tensor(
                    [global_to_local[int(d)] for d in dst[edge_mask].tolist()],
                    dtype=torch.long,
                )
                local_edge_index = torch.stack([local_src, local_dst], dim=0)
                local_edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
            else:
                local_edge_index = torch.zeros((2, 0), dtype=torch.long)
                local_edge_attr = None

            client_data = Data(
                x=x[node_mask],
                edge_index=local_edge_index,
                edge_attr=local_edge_attr,
            )
            partitions.append(client_data)
            self._client_data_sizes.append(int(node_mask.sum()))

        return partitions

    # ------------------------------------------------------------------
    # Cable-based partitioning
    # ------------------------------------------------------------------

    def partition_by_cable(self, cables: list, n_clients: int) -> List[List]:
        """Assign cables to clients using round-robin with load balancing.

        Cables are sorted by length (descending) before assignment so that
        clients receive a balanced total length.

        Args:
            cables: list of cable dicts (must contain 'length_km').
            n_clients: number of clients.

        Returns:
            List of *n_clients* sub-lists, each containing cable dicts.
        """
        n_clients = max(1, n_clients)
        sorted_cables = sorted(
            cables,
            key=lambda c: float(c.get("length_km", 0.0)),
            reverse=True,
        )

        client_loads = [0.0] * n_clients
        client_cables: List[List] = [[] for _ in range(n_clients)]

        for cable in sorted_cables:
            # Assign to least-loaded client
            target = int(np.argmin(client_loads))
            client_cables[target].append(cable)
            client_loads[target] += float(cable.get("length_km", 0.0))

        self._client_data_sizes = [len(c) for c in client_cables]
        return client_cables

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_client_data_sizes(self) -> List[int]:
        """Return the dataset sizes per client from the last partitioning call.

        Returns:
            List of integers (one per client).
        """
        return list(self._client_data_sizes)
