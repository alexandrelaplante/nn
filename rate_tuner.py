from data import DataSet
from gradient_descent import StochasticGradientDescent
from stopping import LastImprovement


# Usage:
#
# rate_tuner = RateTuner(sgd)
# rate_tuner.train_and_tune_rate(
#     training_data=data.training,
#     batch_size=10,
#     learning_rate=0.05,
#     regularization=L2Regularization(lmbda=50.0, n=len(data.training)),
#     data=data,
# )


class RateTuner:
    def __init__(self, sgd: StochasticGradientDescent) -> None:
        self.sgd = sgd

    def train_and_tune_rate(self, learning_rate: float, data: DataSet, *args, **kwargs):
        og_rate = learning_rate
        while learning_rate >= og_rate / 128:
            print("learning rate", learning_rate)
            self.sgd.train(
                learning_rate=learning_rate,
                stopping=LastImprovement(net=self.sgd.network, lookback=3, data=data),
                *args,
                **kwargs
            )
            learning_rate /= 2
