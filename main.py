from evaluate import Evaluator
from gradient_descent import StochasticGradientDescent
from initializers import Normal, Bengio, ScaledNormal
from mnist import MNISTLoader
from network import Network
from layers import Sigmoid, Relu, Linear, Softmax, Input
from cost import Quadratic, CrossEntropy, LogLikelihood
from regularization import L1Regularization, L2Regularization
from stopping import AverageImprovement, Epochs


if __name__ == "__main__":
    data = MNISTLoader.load()
    n = Network(
        layers=[
            (784, Input, Normal),
            (800, Relu, ScaledNormal),
            (10, Sigmoid, Bengio),
        ]
    )
    sgd = StochasticGradientDescent(n, cost=CrossEntropy)

    evalator = Evaluator(n)

    sgd.train(
        training_data=data.training,
        batch_size=10,
        learning_rate=0.1,
        regularization=L2Regularization(lmbda=5.0, n=len(data.training)),
        stopping=Epochs(epochs=300, show_accuracy=False, net=n, data=data),
        # stopping=AverageImprovement(net=n, data=data, threshold=0.001),
    )

    accuracy = evalator.evaluate(data.test)
    print("post-training accuracy", accuracy)
