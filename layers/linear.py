from layers.base import Layer


class Linear(Layer):
    @staticmethod
    def f(x):
        return x

    @staticmethod
    def f_prime(x):
        return 1
