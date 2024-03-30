import numpy as np
from cost import Quadratic
from data import LabeledData

from evaluate import Evaluator
from gradient_descent import StochasticGradientDescent
from mnist import MNISTLoader
from network import Network


class VectorSGD(StochasticGradientDescent):
    """Older and slow implemention where each input is computed separately"""

    def _update(self, batch: list[LabeledData], learning_rate: float) -> None:
        layers = self.network.layers
        batched_delta_w = [np.zeros(shape=layer.w.shape) for layer in layers]
        batched_delta_b = [np.zeros(shape=layer.b.shape) for layer in layers]

        # C_w is partial derivative of cost(data) w.r.t. w_jkl
        # C_b is partial derivative of cost(data) w.r.t. b_jl
        C_w, C_b = self.cost.partials()

        # Skip the input layer
        for l in range(1, len(layers)):
            step_size = learning_rate / len(batch)
            batched_delta_w[l] -= step_size * sum(C_w(data, l) for data in batch)
            batched_delta_b[l] -= step_size * sum(C_b(data, l) for data in batch)

        # Update network
        for l, layer in enumerate(layers):
            layer.w += batched_delta_w[l]
            layer.b += batched_delta_b[l]


if __name__ == "__main__":
    from sigmoid import SigmoidLayer
    from relu import ReluLayer

    data = MNISTLoader.load()
    n = Network(sizes=[784, 10, 10], layer_classes=[SigmoidLayer, SigmoidLayer])
    sgd = VectorSGD(n, cost=Quadratic)

    evalator = Evaluator(n)
    accuracy = evalator.evaluate(data.test)
    print("pre-training accuracy", accuracy)

    sgd.train(
        training_data=data.training,
        epochs=3,
        batch_size=100,
        learning_rate=6.0,
    )

    accuracy = evalator.evaluate(data.test)
    print("post-training accuracy", accuracy)
