class Node:
    def __init__(self, left, right):
        self.left, self.right = left, right
        self.ls, self.rs = None, None

        self.value = 0
        self.tag = 0

    def mid(self):
        return (self.left + self.right) // 2

    def length(self):
        return self.right - self.left + 1

    def update(self, delt):
        self.tag += delt
        self.value += delt * self.length()

    def push_up(self):
        self.value = self.ls.value + self.rs.value

    def push_down(self):
        delt = self.tag
        self.ls.update(delt), self.rs.update(delt)
        self.tag = 0


class SegmentTree:
    def __init__(self, array_in):
        n = len(array_in)
        array = [0] + array_in

        def build(left, right):
            node = Node(left, right)

            if left == right:
                node.value = array[left]
            else:
                m = node.mid()

                node.ls = build(left, m)
                node.rs = build(m + 1, right)

                node.push_up()

            return node

        self.root = build(1, n)

    def modify(self, left, right, delt):

        def dfs(node):
            if left <= node.left and node.right <= right:
                node.update(delt)
            else:
                node.push_down()

                m = node.mid()

                if left <= m:
                    dfs(node.ls)
                if right > m:
                    dfs(node.rs)

                node.push_up()

        dfs(self.root)

    def query(self, left, right):

        def dfs(node):
            if left <= node.left and node.right <= right:
                return node.value

            node.push_down()

            res = 0
            m = node.mid()

            if left <= m:
                res += dfs(node.ls)
            if m < right:
                res += dfs(node.rs)

            return res

        return dfs(self.root)