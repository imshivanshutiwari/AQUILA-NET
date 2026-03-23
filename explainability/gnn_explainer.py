from __future__ import annotations

from typing import List

import torch
import numpy as np
from torch_geometric.explain import Explainer, GNNExplainer


class AQUILAGNNExplainer:
    """Node-level GNN explanations via torch_geometric's GNNExplainer."""

    def __init__(self, model: torch.nn.Module, algorithm=None) -> None:
        self.model = model
        algo = algorithm if algorithm is not None else GNNExplainer(epochs=200)
        self.explainer = Explainer(
            model=model,
            algorithm=algo,
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="binary_classification",
                task_level="node",
                return_type="raw",
            ),
        )

    def explain_node(self, node_idx: int, data) -> dict:
        """Generate an explanation for a single node.

        Args:
            node_idx: Index of the node to explain.
            data: torch_geometric Data object.

        Returns:
            Dict with keys 'edge_mask', 'node_mask', 'top5_edge_indices'.
        """
        edge_attr = (
            data.edge_attr
            if hasattr(data, "edge_attr") and data.edge_attr is not None
            else None
        )

        explanation = self.explainer(
            data.x,
            data.edge_index,
            index=node_idx,
            edge_attr=edge_attr,
        )

        if hasattr(explanation, "edge_mask") and explanation.edge_mask is not None:
            edge_mask = explanation.edge_mask
        else:
            edge_mask = torch.ones(data.edge_index.shape[1])

        if hasattr(explanation, "node_mask") and explanation.node_mask is not None:
            node_mask = explanation.node_mask
        else:
            node_mask = torch.ones(data.x.shape[0], 1)

        top5_edges = torch.topk(edge_mask, min(5, len(edge_mask))).indices

        return {
            "edge_mask": edge_mask.detach().cpu().numpy(),
            "node_mask": node_mask.detach().cpu().numpy(),
            "top5_edge_indices": top5_edges.tolist(),
        }

    def explain_batch(self, data, anomaly_nodes: List[int]) -> List[dict]:
        """Explain all nodes in *anomaly_nodes*.

        Args:
            data: torch_geometric Data object.
            anomaly_nodes: List of node indices to explain.

        Returns:
            List of explanation dicts, one per node.
        """
        return [self.explain_node(n, data) for n in anomaly_nodes]
