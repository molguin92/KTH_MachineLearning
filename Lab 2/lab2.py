import math
from pprint import PrettyPrinter
from typing import Callable, List, Tuple

import numpy
from cvxopt.base import matrix
from cvxopt.solvers import qp
from matplotlib import pylab

import generate

FLOAT_THRESHOLD = 10.0 ** (-5)
print = PrettyPrinter(indent=4).pprint


def dot_product(x: tuple, y: tuple) -> float:
    assert len(x) == len(y)
    res = 0.0
    for i, j in zip(x, y):
        res += i * j

    return res


def poly_kernel(x: tuple, y: tuple, d: int) -> float:
    assert len(x) == len(y)
    dot = dot_product(x, y)
    result = math.pow(dot + 1.0, d)
    return result


def linear_kernel(x: tuple, y: tuple) -> float:
    return poly_kernel(x, y, 1)


def radial_kernel(x: tuple, y: tuple, gamma: float) -> float:
    assert len(x) == len(y)
    sum = 0.0
    for i, j in zip(x, y):
        sum += math.pow(i - j, 2)

    return math.exp(-1.0 * gamma * sum)


def sigmoid_kernel(x: tuple, y: tuple, k: float, delta: float) -> float:
    assert len(x) == len(y)
    dot = dot_product(x, y)
    res = math.tanh((k * dot) - delta)
    return res


def train(training_data: List[Tuple[float, float, float]],
          kernel: Callable = linear_kernel, slack: bool = False,
          C: float = None) -> Callable:
    if slack:
        assert C

    X = [(e[0], e[1]) for e in training_data]
    T = [e[2] for e in training_data]
    N = len(training_data)

    # build P
    P = numpy.empty(shape=(N, N))
    for i in range(N):
        for j in range(N):
            P[i][j] = T[i] * T[j] * kernel(X[i], X[j])

    # build q
    q = numpy.ones(N) * -1

    # build h
    h = None
    if slack:
        h = numpy.empty(2 * N)
        for i in range(N):
            h[i] = 0
        for i in range(N, 2 * N):
            h[i] = C
    else:
        h = numpy.zeros(N)

    # build G
    G = numpy.zeros(shape=(2 * N, N)) if slack else numpy.zeros(shape=(N, N))
    for i, j in zip(range(N), range(N)):
        G[i][j] = -1

    if slack:
        for i, j in zip(range(N, 2 * N), range(N)):
            G[i][j] = 1

    r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
    alpha = list(r['x'])

    # get all non-zero alphas and corresponding points and classes
    alpha_nz = []
    for i in range(len(alpha)):
        if alpha[i] >= FLOAT_THRESHOLD:
            alpha_nz.append((alpha[i], X[i], T[i]))

    def indicator(nX):
        sum = 0.0
        for (a, x, t) in alpha_nz:
            sum += a * t * kernel(nX, x)
        return sum

    return indicator


if __name__ == '__main__':
    data = generate.generate(points=50)
    indicator = train(data,
                      kernel=lambda x, y: radial_kernel(x, y, 3),
                      slack=True, C=5)

    xrange = numpy.arange(-8, 8, 0.05)
    yrange = numpy.arange(-8, 8, 0.05)

    p_grid = [[indicator((x, y)) for y in yrange] for x in xrange]
    grid = matrix(p_grid)

    # pylab.hold(True)
    pylab.plot([p[0] for p in data if p[2] > 0],
               [p[1] for p in data if p[2] > 0], 'bo')
    pylab.plot([p[0] for p in data if p[2] < 0],
               [p[1] for p in data if p[2] < 0], 'ro')
    pylab.contour(xrange, yrange, grid, (-1.0, 0.0, 1.0),
                  colors=('red', 'black', 'blue'), linewidths=(1, 1, 1))
    pylab.show()
