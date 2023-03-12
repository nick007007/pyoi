class ModCalculation:
    def __init__(self, mod=0):
        self.mod = mod

    def get_mod(self, a):
        return a % self.mod if self.mod else a

    def _add(self, a, b):
        return (a + b) % self.mod if self.mod else a + b

    def add(self, *args):
        s = 0
        for i in args:
            s = self._add(s, i)
        return s

    def sum(self, iterable):
        return self.add(*iterable)

    def _mul(self, a, b):
        return (a * b) % self.mod if self.mod else a * b

    def mul(self, *args):
        res = 1
        for i in args:
            res = self._mul(res, i)
        return res

    def quick_power(self, a, n):
        res = 1

        while n:
            if n & 1:
                res = self.mul(res, a)

            a = self.mul(a, a)
            n >>= 1

        return res


class LinearSpace(ModCalculation):
    def __init__(self, dim, mod=0):
        super().__init__(mod)
        self.dim = dim

        self.E = [[0] * dim for _ in range(dim)]
        for i in range(dim):
            self.E[i][i] = 1

    def axb(self, a, b):
        return self.sum(self.mul(i, j) for i, j in zip(a, b))

    def axA(self, a, A):
        return [self.sum(self.mul(a[i], A[i][j]) for i in range(self.dim)) for j in range(self.dim)]

    def AxB(self, A, B):
        return [self.axA(a, B) for a in A]

    def quick_power(self, A, n):
        res = self.E

        B = A
        while n:
            if n & 1:
                res = self.AxB(res, B)

            B = self.AxB(B, B)
            n >>= 1

        return res


def break_factor(n):
    sqrt = int(n ** 0.5)
    factors = [sqrt]

    if n // sqrt != sqrt:
        factors.append(n // sqrt)

    for i in range(1, sqrt):
        if n % i == 0:
            factors.append(i)
            factors.append(n // i)

    return factors


def linear_sieve(n):            # primes in [2, n]
    is_prime, primes = [True] * (n + 1), []

    for p in range(2, n + 1):
        if is_prime[p]:
            primes.append(p)

            for i in range(p * p, n + 1, p):
                is_prime[i] = False

    return primes


def is_prime(n):
    sqrt = int(n ** 0.5)
    primes = linear_sieve(sqrt)

    for i in primes:
        if n % i == 0:
            return False

    return True


def factorization(n):
    p = linear_sieve(n // 2)
    factors = {}

    def push(p):
        if p not in factors:
            factors[p] = 0

        factors[p] += 1

    for i in p:
        while n % i == 0:
            push(i)
            n //= i

        if n == 1:
            break

    if not factors and n > 1:
        push(n)

    return factors