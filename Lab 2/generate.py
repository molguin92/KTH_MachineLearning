import random
from math import ceil, floor
from typing import List, Tuple


def generate(points: float = 10) -> List[Tuple[float, float, float]]:
    points = points * 1.0
    classA = [(random.normalvariate(-1.5, 1),
               random.normalvariate(0.5, 1),
               1.0) for i in range(int(floor(points / 2)))] + \
             [(random.normalvariate(1.5, 1),
               random.normalvariate(0.5, 1),
               1.0) for i in range(int(ceil(points / 2)))]

    classB = [(random.normalvariate(0.0, 0.5),
               random.normalvariate(-2.0, 2.0), -1.0)
              for i in range(int(floor(points)))]

    data = classA + classB
    random.shuffle(data)

    return data
