from tqdm import tqdm


class ProgressBar:
    def __init__(self, total: int) -> None:
        self.t = tqdm(
            total=total,
            unit=" samples",
            unit_scale=True,
            postfix={"epochs": 1, "accuracy": "0%"},
            smoothing=0.1,
        )

    def update(self, n: int) -> None:
        self.t.update(n=n)

    def set_postfix(self, epoch: int, accuracy: float):
        postfix = {"epochs": epoch}

        if accuracy is not None:
            postfix["accuracy"] = f"{100*accuracy}%"
        self.t.set_postfix(postfix)
