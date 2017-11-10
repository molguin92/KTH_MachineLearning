import random
from math import ceil, floor
from typing import List, Tuple

DATASET = List[Tuple[float, float, float]]


def generate(points: float = 10) -> DATASET:
    points = points * 1.0
    classA = [(random.normalvariate(-1.5, 1),
               random.normalvariate(0.5, 1),
               1.0) for i in range(int(floor(points / 2)))] + \
             [(random.normalvariate(1.5, 1),
               random.normalvariate(0.5, 1),
               1.0) for i in range(int(ceil(points / 2)))]

    classB = [(random.normalvariate(0.0, 0.5),
               random.normalvariate(-0.5, 0.5), -1.0)
              for i in range(int(floor(points)))]

    data = classA + classB
    random.shuffle(data)

    return data


def generate3(points: float = 10) -> Tuple[DATASET, DATASET, DATASET]:
    points = points * 1.0

    # high separation
    classA = [(random.normalvariate(-1.5, 1),
               random.normalvariate(-1.5, 1),
               1.0) for i in range(int(floor(points / 2)))] + \
             [(random.normalvariate(-1.5, 1),
               random.normalvariate(-1.5, 1),
               1.0) for i in range(int(ceil(points / 2)))]

    classB = [(random.normalvariate(1.5, -1),
               random.normalvariate(1.5, -1), -1.0)
              for i in range(int(floor(points)))]

    dataset_A = classA + classB
    random.shuffle(dataset_A)

    # low separation
    classA = [(random.normalvariate(0, 1),
               random.normalvariate(0, 1),
               1.0) for i in range(int(floor(points / 2)))] + \
             [(random.normalvariate(0.5, 1.5),
               random.normalvariate(0.5, 1.5),
               1.0) for i in range(int(ceil(points / 2)))]

    classB = [(random.normalvariate(0.75, 1.25),
               random.normalvariate(0.75, 1.25), -1.0)
              for i in range(int(floor(points)))]

    dataset_B = classA + classB
    random.shuffle(dataset_B)

    # intermediate separation
    classA = [(random.normalvariate(-1.5, 1),
               random.normalvariate(0.5, 1),
               1.0) for i in range(int(floor(points / 2)))] + \
             [(random.normalvariate(1.5, 1),
               random.normalvariate(0.5, 1),
               1.0) for i in range(int(ceil(points / 2)))]

    classB = [(random.normalvariate(0.0, 0.5),
               random.normalvariate(-0.5, 0.5), -1.0)
              for i in range(int(floor(points)))]

    dataset_C = classA + classB
    random.shuffle(dataset_C)

    return (dataset_A, dataset_B, dataset_C)
