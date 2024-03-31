from layers.base import Layer
import np


# def softmax(z: np.ndarray) -> np.ndarray:
#     return np.exp(z) / np.sum(np.exp(z))


def softmax(z: np.ndarray) -> np.ndarray:
    # Taken from online to avoid overflows
    exps = np.exp(z - np.amax(z))
    return exps / np.sum(exps, axis=1, keepdims=True)


def softmax_grad_vec(s):
    # s = z.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def softmax_prime(s: np.ndarray) -> np.ndarray:
    # cols = []
    # for i in range(s.shape[1]):
    #     print(s[:][i].shape)
    #     cols.append(softmax_grad_vec(s[:][i]))
    # print(cols[0].shape)
    # res = np.column_stack(cols)
    # print("s.shape", s.shape)
    # print("res.shape", res.shape)
    # return res
    raise NotImplementedError


class Softmax(Layer):
    @staticmethod
    def f(x):
        return softmax(x)

    @staticmethod
    def f_prime(x):
        # raise NotImplementedError
        return softmax_prime(x)
