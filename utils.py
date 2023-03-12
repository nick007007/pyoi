class Cache:
    def __init__(self):
        from sys import setrecursionlimit
        from functools import lru_cache

        setrecursionlimit(int(1e6))

        self.lru_cache = lru_cache

    def func(self):
        return self.lru_cache(maxsize=None)     # return key is function


def num_to_int(s):
    try:
        return int(s)
    except:
        return s


def extend0_matrix_input(n, m):
    return [[0] * (m + 1)] + [[0] + [int(x) for x in input().split()] for _ in range(n)]


def around(x, y):
    yield x + 1, y
    yield x - 1, y
    yield x, y + 1
    yield x, y - 1


def clockwise_reshape(list, index, reverse=False):
    step = -1 if reverse else 1

    return list[index:: step] + list[: index: step]