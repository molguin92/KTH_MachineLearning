from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy, random, math
from matplotlib import pylab
import generate

from typing import List, Callable, Tuple

FLOAT_THRESHOLD = 10.0 ** (-5)


def poly_kernel(x: tuple, y: tuple, d: int) -> float:
    assert len(x) == len(y)

    sum = 0.0
    for i, j in zip(x, y):
        sum += i * j

    result = math.pow(sum + 1.0, d)
    return result


def linear_kernel(x: tuple, y: tuple) -> float:
    return poly_kernel(x, y, 1)


def radial_kernel(x: tuple, y: tuple, gamma: float) -> float:
    assert len(x) == len(y)

    sum = 0.0
    for i, j in zip(x, y):
        sum += math.pow(i - j, 2)

    return math.exp(-1.0 * gamma * sum)


def train(training_data: List[Tuple[float, float, float]],
          kernel: Callable = linear_kernel) -> Callable:
    X = [(e[0], e[1]) for e in training_data]
    T = [e[2] for e in training_data]
    N = len(training_data)

    # build P
    P = numpy.empty(shape=(N, N))
    for i in range(N):
        for j in range(N):
            P[i][j] = T[i] * T[j] * kernel(X[i], X[j])

    print('P: ', P)

    # build q
    q = numpy.ones(N) * -1
    print('q: ', q)

    # build h
    h = numpy.zeros(N)
    print('h: ', h)

    # build G
    G = numpy.zeros(shape=(N, N))
    for i, j in zip(range(N), range(N)):
        G[i][j] = -1

    r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
    alpha = list(r['x'])

    # get all non-zero alphas and corresponding points and classes
    alpha_nz = []
    for i in range(N):
        if alpha[i] >= FLOAT_THRESHOLD:
            alpha_nz.append((alpha[i], X[i], T[i]))

    def indicator(nX):
        sum = 0.0
        for (a, x, t) in alpha_nz:
            sum += a * t * kernel(nX, x)
        return sum

    return indicator


if __name__ == '__main__':
    data = generate.generate()
    indicator = train(data, kernel=lambda x, y: radial_kernel(x, y, 1))

    xrange = numpy.arange(-4, 4, 0.05)
    yrange = numpy.arange(-4, 4, 0.05)

    p_grid = [[indicator((x, y)) for y in yrange] for x in xrange]
    print(p_grid)
    grid = matrix(p_grid)

    pylab.hold(True)
    pylab.plot([p[0] for p in data if p[2] > 0],
               [p[1] for p in data if p[2] > 0], 'bo')
    pylab.plot([p[0] for p in data if p[2] < 0],
               [p[1] for p in data if p[2] < 0], 'ro')
    pylab.contour(xrange, yrange, grid, (-1.0, 0.0, 1.0),
                  colors=('red', 'black', 'blue'), linewidths=(1, 1, 1))
    pylab.show()
