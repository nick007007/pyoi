class PrefixArea:
    def __init__(self, matrix):                 # matrix extend 0
        n = len(matrix)
        m = len(matrix[0])

        self.area = [[0] * m for _ in range(n)]

        for i in range(1, n):
            for j in range(1, m):
                self.area[i][j] = matrix[i][j] + self.two_area(i - 1, j - 1, i, j)

    def two_area(self, x1, y1, x2, y2):         # x1 <= x2, y1 <= y2
        return self.area[x1][y2] + self.area[x2][y1] - self.area[x1][y1]

    def query(self, x1, y1, x2, y2):
        return self.area[x2][y2] - self.two_area(x1 - 1, y1 - 1, x2, y2)