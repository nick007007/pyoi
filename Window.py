class Window:
    from collections import deque

    def __init__(self):
        self.queue = self.deque()


    def __len__(self):
        return len(self.queue)

    def push(self, x):
        self.queue.append(x)


    def pop(self):
        head = self.queue.popleft()


    def true(self):

        return


def LongestSub(l):
    window = Window()

    M = 0
    for i in l:
        window.push(i)                  # 初始时为真，只有判断为真的时候才会加入新元素，目的是越长越好

        while not window.true():        # 判断为假时不断弹出队列头元素，直到判断为真
            window.pop()

        M = max(M, len(window))         # 在判断为真时更新最大值

    return M


def ShortestSub(l):
    window = Window()

    m = float('inf')
    for i in l:
        window.push(i)                  # 初始时为假, 只有判断为假的时候才会加入新元素，目的是越短越好

        while window.true():            # 判断为真时不断更新最小值，弹出队列头元素，直到判断为假
            m = min(m, len(window))

            window.pop()

    return m                            # 注意是否存在窗口为真的情况


class WindowCounter:
    def __init__(self, left):           # 窗口最左端初始位置
        self.left = left
        self.right = left - 1

    def __len__(self):
        return self.right - self.left + 1

    def push(self):
        self.right += 1

    def pop(self):
        self.left += 1

    def true(self):

        return
