from evaluate import Evaluator
from gradient_descent import StochasticGradientDescent
from mnist import MNISTLoader
from network import Network
from layers import SigmoidLayer, ReluLayer, LinearLayer, SoftmaxLayer
from cost import Quadratic, CrossEntropy, LogLikelihood
from stopping import AverageImprovement, Epochs


if __name__ == "__main__":
    data = MNISTLoader.load()
    n = Network(
        sizes=[784, 30, 10],
        layer_classes=[SigmoidLayer, SigmoidLayer],
    )
    sgd = StochasticGradientDescent(n, cost=Quadratic)

    evalator = Evaluator(n)
    accuracy = evalator.evaluate(data.test)
    print("pre-training accuracy", accuracy)

    sgd.train(
        training_data=data.training,
        batch_size=10,
        learning_rate=0.5,
        reg_param=5.0,
        stopping=Epochs(net=n, data=data, epochs=30, show_accuracy=True),
        # stopping=AverageImprovement(net=n, data=data, threshold=0.001),
    )

    accuracy = evalator.evaluate(data.test)
    print("post-training accuracy", accuracy)
