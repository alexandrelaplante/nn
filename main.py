from evaluate import Evaluator
from gradient_descent import StochasticGradientDescent
from initializers import Normal, Bengio, ScaledNormal
from mnist import MNISTLoader
from network import Network
from layers import Sigmoid, Relu, Linear, Softmax, Tanh, Input
from cost import Quadratic, CrossEntropy, LogLikelihood
from rate_tuner import RateTuner
from regularization import L1Regularization, L2Regularization
from stopping import AverageImprovement, Epochs, LastImprovement


if __name__ == "__main__":
    data = MNISTLoader.load()
    n = Network(
        layers=[
            (784, Input, Normal),
            (100, Relu, ScaledNormal),
            (10, Relu, ScaledNormal),
        ]
    )
    sgd = StochasticGradientDescent(n, cost=CrossEntropy)

    evalator = Evaluator(n)

    sgd.train(
        training_data=data.training,
        batch_size=100,
        learning_rate=0.05,
        regularization=L2Regularization(lmbda=50.0, n=len(data.training)),
        stopping=Epochs(epochs=3, show_accuracy=False, net=n, data=data),
        momentum_coefficient=0.8,
        # stopping=AverageImprovement(net=n, data=data, threshold=0.001),
        # stopping=LastImprovement(net=n, lookback=10, data=data),
    )

    accuracy = evalator.evaluate(data.test)
    print("post-training accuracy", accuracy)
