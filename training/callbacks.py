"""Small training-loop callbacks."""

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int
    best: float = float("-inf")
    stale_epochs: int = 0

    def update(self, value: float) -> bool:
        if value > self.best:
            self.best = value
            self.stale_epochs = 0
            return False
        self.stale_epochs += 1
        return self.stale_epochs >= self.patience
