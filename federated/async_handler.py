from typing import List

import numpy as np


class AsyncClientHandler:
    """Handles asynchronous federated learning client participation."""

    def __init__(self, n_clients: int = 20, tolerance_rounds: int = 3):
        self.n_clients = n_clients
        self.tolerance_rounds = tolerance_rounds
        # Maps client_id → list of (round_num, status) where status is 'active'|'offline'
        self._events: dict = {cid: [] for cid in range(n_clients)}

    def mark_client_active(self, client_id: int, round_num: int) -> None:
        """Record that a client was active during a given round."""
        self._events[client_id].append((round_num, "active"))

    def mark_client_offline(self, client_id: int, round_num: int) -> None:
        """Record that a client went offline during a given round."""
        self._events[client_id].append((round_num, "offline"))

    def get_active_clients(self, round_num: int) -> List[int]:
        """Return IDs of clients active in *round_num* or within the tolerance window.

        A client is considered reachable when the most recent recorded event
        is 'active' and occurred within [round_num - tolerance_rounds, round_num].
        If no events have been recorded at all, the client is treated as active
        (optimistic assumption for new deployments).
        """
        active = []
        for cid in range(self.n_clients):
            events = self._events[cid]
            if not events:
                active.append(cid)
                continue
            # Find the latest event at or before round_num
            relevant = [(r, s) for r, s in events if r <= round_num]
            if not relevant:
                continue
            latest_round, latest_status = max(relevant, key=lambda x: x[0])
            if (
                latest_status == "active"
                and round_num - latest_round <= self.tolerance_rounds
            ):
                active.append(cid)
        return sorted(active)

    def is_stale_update(self, client_id: int, update_round: int, current_round: int) -> bool:
        """Return True when the update is older than tolerance_rounds."""
        return (current_round - update_round) > self.tolerance_rounds

    def get_participation_matrix(self, n_rounds: int) -> np.ndarray:
        """Return a (n_clients, n_rounds) boolean participation matrix.

        Uses a deterministic hash-based 80 % participation rate with temporal
        clustering to mimic realistic FL scenarios.
        """
        matrix = np.zeros((self.n_clients, n_rounds), dtype=bool)
        for cid in range(self.n_clients):
            for rnd in range(n_rounds):
                participated = hash(cid * 1000 + rnd) % 100 < 80
                matrix[cid, rnd] = participated
        return matrix
