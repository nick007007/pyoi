class MonotonicQueue:
    from collections import deque

    def __init__(self, type, sub=[]):
        self.type = type == 'decrease'

        self.queue = self.deque()

        for i in sub:
            self.push(i)

    def head(self):
        return self.queue[0]

    def is_monotonic(self, x):
        tail = self.queue[-1]

        return tail >= x if self.type else tail <= x

    def push(self, x):
        while self.queue and not self.is_monotonic(x):
            self.queue.pop()

        self.queue.append(x)

    def pop(self, value):
        if self.head() == value:
            self.queue.popleft()