import numpy as np
from pprint import pprint

class Perceptron:
    def __init__(self, w, b, name=None):
        self.w = np.array(w)
        self.b = b
        self.name = name

    def apply(self, x):
        x = np.array(x)
        w = self.w
        b = self.b
        return int(np.dot(w, x) + b > 0)
    
    def __repr__(self):
        return self.name


class Circuit:
    def __init__(self, neurons, connections):
        self.neurons = neurons
        self.connections = connections
    
    def apply(self, x):
        outputs = x.copy()
        for p in self.neurons:
            inputs = [
                outputs[_from]
                for _from, _to in self.connections
                if _to == p
            ]
            outputs[p] = p.apply(inputs)
        return outputs

if __name__ == '__main__':
    x1 = Perceptron(name='x1', w=None, b=None)
    x2 = Perceptron(name='x2', w=None, b=None)
    p1 = Perceptron(name='p1', w=[-2,-2], b=3)
    p2 = Perceptron(name='p2', w=[-2,-2], b=3)
    p3 = Perceptron(name='p3', w=[-2,-2], b=3)
    p4 = Perceptron(name='p4', w=[-4], b=3)
    p5 = Perceptron(name='p5', w=[-2,-2], b=3)

    connections = [
        (0, p1),
        (x1, p2),
        (x2, p1),
        (x2, p3),
        (p1, p2),
        (p1, p3),
        (p1, p4),
        (p2, p5),
        (p3, p5),
    ]

    circuit = Circuit([p1, p2, p3, p4, p5], connections)
    answer = circuit.apply({x1: 1, x2: 1})
    pprint(answer)
