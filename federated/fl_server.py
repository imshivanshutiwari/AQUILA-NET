from typing import Dict, List, Optional, Tuple

import torch


class FederatedServer:
    """Central server coordinating FedAvg federated learning rounds."""

    def __init__(
        self,
        model: torch.nn.Module,
        n_clients: int = 20,
        config: Optional[dict] = None,
    ):
        self.model = model
        self.n_clients = n_clients
        self.config = config or {}
        self.round_metrics: List[dict] = []
        self.global_round: int = 0

    def aggregate_weights(self, client_updates: List[Tuple[dict, int, float]]) -> dict:
        """FedAvg: compute a sample-weighted average of client state dicts.

        Args:
            client_updates: list of (state_dict, n_samples, loss)

        Returns:
            Averaged state dict.
        """
        total_samples = sum(n for _, n, _ in client_updates)
        if total_samples == 0:
            # Fall back to uniform weighting if all n_samples are zero
            total_samples = len(client_updates)
            weights = [1.0 / total_samples] * len(client_updates)
        else:
            weights = [n / total_samples for _, n, _ in client_updates]

        averaged: dict = {}
        for key in client_updates[0][0].keys():
            stacked = torch.stack(
                [sd[key].float() * w for (sd, _, _), w in zip(client_updates, weights)]
            )
            averaged[key] = stacked.sum(dim=0).to(client_updates[0][0][key].dtype)
        return averaged

    def select_clients(self, fraction: float = 0.8) -> List[int]:
        """Deterministically select a fraction of clients for the current round.

        Clients are ranked by hash(global_round * 10000 + client_id) % 100 and
        the top-n_select by ascending hash value are chosen.
        """
        n_select = max(1, int(self.n_clients * fraction))
        scores = [
            (hash(self.global_round * 10000 + cid) % 100, cid)
            for cid in range(self.n_clients)
        ]
        selected = sorted(scores)[:n_select]
        return sorted(cid for _, cid in selected)

    def run_round(self, clients: list, fraction: float = 0.8) -> dict:
        """Execute one federated round.

        Steps:
        1. Select clients.
        2. Broadcast current global weights to each selected client.
        3. Collect local training results.
        4. Aggregate via FedAvg and update the global model.

        Args:
            clients: list of FederatedClient objects indexed by client_id.
            fraction: fraction of clients to involve in this round.

        Returns:
            Metrics dict: {round, loss, n_participants, epsilon}.
        """
        selected_ids = self.select_clients(fraction)
        global_weights = self.get_global_model()

        client_updates: List[Tuple[dict, int, float]] = []
        for cid in selected_ids:
            client = clients[cid]
            state_dict, n_samples, loss = client.local_train(global_weights)
            client_updates.append((state_dict, n_samples, loss))

        averaged = self.aggregate_weights(client_updates)
        self.model.load_state_dict(averaged)

        avg_loss = (
            sum(loss for _, _, loss in client_updates) / len(client_updates)
            if client_updates
            else float("nan")
        )

        metrics = {
            "round": self.global_round,
            "loss": avg_loss,
            "n_participants": len(selected_ids),
            "epsilon": None,
        }
        self.round_metrics.append(metrics)
        self.global_round += 1
        return metrics

    def get_global_model(self) -> dict:
        """Return the current global model state dict."""
        return self.model.state_dict()
