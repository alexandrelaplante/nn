from layers.base import Layer


class Input(Layer):
    @staticmethod
    def f(x):
        raise NotImplementedError

    @staticmethod
    def f_prime(x):
        raise NotImplementedError
