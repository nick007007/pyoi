from collections import deque
from heapq import heappush, heappop

from number_theory import LinearSpace
from UFS import UFS


class AdjSet:
    def __init__(self, n):
        self.n = n
        self.in_degree = [0] * (n + 1)

        self.set = [set() for _ in range(n + 1)]  # default 0 and 1 to n vertex

    def __getitem__(self, item):
        return self.set[item]

    def add_edge(self, u, v):
        if v not in self[u]:
            self[u].add(v)
            self.in_degree[v] += 1

    def topsort(self):
        n = self.n
        in_degree = self.in_degree

        visited = [False] * (n + 1)
        res = []
        q = deque()

        def push(v):
            if in_degree[v] == 0:
                if visited[v]:
                    return True                     # exist cycle

                visited[v] = True
                q.append(v)
                res.append(v)

        for u in range(1, n + 1):
            push(u)

        while q:
            u = q.popleft()

            for v in self[u]:
                in_degree[v] -= 1

                if push(v):
                    return False

        if len(res) == n:                           # graph connected
            return res


class AdjList:
    def __init__(self, n):
        self.n = n
        self.list = [{} for _ in range(n + 1)]  # default 0 and 1 to n vertex

    def __getitem__(self, item):
        return self.list[item]

    def add_edge(self, u, v, w):
        adj = self[u]

        adj[v] = min(adj[v], w) if v in adj else w

    def dijkstra(self, a, b):
        n = self.n
        dist = [float('inf')] * (n + 1)

        tree = set()
        q = []

        def push(d_min, v):
            if dist[v] > d_min:
                dist[v] = d_min
                heappush(q, (d_min, v))

        push(0, a)

        while q:
            d, v = heappop(q)

            if v == b:
                return d

            if v not in tree:
                tree.add(v)

                for u, w in self[v].items():
                    d_min = d + w
                    push(d_min, u)

        return -1


class AdjMatrix:
    def __init__(self, n):
        self.n = n                                  # default 0 and 1 to n vertex
        self.matrix = [[float('inf')] * (n + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            self[i][i] = 0

    def __getitem__(self, item):
        return self.matrix[item]

    def floyd(self):
        n = self.n

        for i in range(1, n + 1):
            for j in range(1, n + 1):
                for k in range(1, n + 1):
                    self[j][k] = min(self[j][k], self[j][i] + self[i][k])

    def prim(self):
        n = self.n
        size = 0

        dist = [float('inf')] * (n + 1)
        mst = set()

        def update(u):
            mst.add(u)
            dist[u] = 0

            for v in range(1, n + 1):
                dist[v] = min(dist[v], self[u][v])

        def next():
            u = 0

            for i in range(1, n + 1):
                if i not in mst:
                    if dist[i] < dist[u]:
                        u = i

            return u

        update(1)

        for _ in range(n - 1):
            u = next()

            if u == 0:
                return

            size += dist[u]
            update(u)

        return size


class SplitAdjMatrix:
    def __init__(self, vertex_num, max_dist):       # vertex_num from 0 to vertex_num - 1
        self.vertex_num = vertex_num
        self.max_dist = max_dist

        self.size = vertex_num * max_dist
        self.matrix = [[0] * self.size for _ in range(self.size)]

        for i in range(vertex_num):
            for j in range(1, max_dist):
                front = self.get_vertex(i, j - 1)
                back = self.get_vertex(i, j)

                self.matrix[front][back] = 1

    def get_vertex(self, i, j):
        return i * self.max_dist + j

    def add_edge(self, a, b, w):
        if w:
            front = self.get_vertex(a, w - 1)
            back = self.get_vertex(b, 0)

            self.matrix[front][back] = 1

    def calc_graph(self, steps, mod=0):
        space = LinearSpace(self.size, mod)

        self.matrix = space.quick_power(self.matrix, steps)

    def calc_path(self, a, b):
        front = self.get_vertex(a, 0)
        back = self.get_vertex(b, 0)

        return self.matrix[front][back]


class EdgeSet:
    def __init__(self, n):
        self.n = n
        self.edges = []

    def add(self, u, v, w):
        if u > v:
            u, v = v, u
        pair = (u, v)

        self.edges.append((w, pair))

    def kruskal(self):
        n = self.n
        ufs = UFS(n)

        size = 0
        edges = sorted(self.edges)

        for w, pair in edges:
            if not ufs.is_connected(*pair):
                ufs.union(*pair)

                size += w

        if ufs.get_size(1) < n:
            return

        return size