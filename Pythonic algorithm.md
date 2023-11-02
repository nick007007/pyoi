# Pythonic algorithm

# 计算机基础

## Mbps、KBps、MB、kb、B、b

1B(字节) = 8b(位)

1MB = 1024KB				1KB = 1024B

1Mbps = 1000Kbps		1Kbps = 1000bps

1MBps = 8Mbps

```python
>>> MBtoKB = lambda x: 1024 * x
... KBtoB = lambda x: 1024 * x
... Btob = lambda x: 8 * x
... MB = 256
... KB = MBtoKB(MB)
... B = KBtoB(KB)
... b = Btob(B)
... n = b // 32
n
67108864
```

## 数据范围反推算法复杂度以及算法内容

时间限制1秒或2秒。在这种情况下，C++代码中的操作次数控制在 10^7∼10^8为最佳。

下面给出在不同数据范围下，代码的时间复杂度和算法该如何选择：

### n ≤ 30

**指数级别**, dfs+剪枝，状态压缩dp

### n ≤ 100

**O(n^3)**，floyd，dp，高斯消元

### n ≤ 1000

**O(n^2)，O(n^2logn)**，dp，二分，朴素版Dijkstra、朴素版Prim、Bellman-Ford

### n ≤ 10000

**O(n∗sqrt(n))**，块状链表、分块、莫队

### n ≤ 100000

**O(nlogn)** => 各种sort，线段树、树状数组、set/map、heap、拓扑排序、dijkstra+heap、prim+heap、Kruskal、spfa、求凸包、求半平面交、二分、CDQ分治、整体二分、后缀数组、树链剖分、动态树

### n≤1000000

**O(n)**, 以及常数较小的 O(nlogn) 算法 => 单调队列、 hash、双指针扫描、并查集，kmp、AC自动机，常数比较小的 O(nlogn) 的做法：sort、树状数组、heap、dijkstra、spfa

### n ≤ 10000000

**O(n)**，双指针扫描、kmp、AC自动机、线性筛素数

### n ≤ $10^9$

**O(sqrt(n))**，判断质数

### n ≤ $10^{18}$

**O(logn)**，最大公约数，快速幂，数位DP

### n ≤ $10^{1000}$

**O((logn)^2)**，高精度加减乘除

### n ≤ $10^{100000}$

**O(logk×loglogk)，k表示位数**，高精度加减、FFT/NTT



# 基本知识

## Python运算符

0为False，非0为True

![image-20220319090755721](https://s2.loli.net/2022/03/19/ir6LMUxNgE1OuA7.png)

## 内置函数

### 判断类型

```python
type(1.0)
Out[2]: float
type(3) == int
Out[3]: True
```

### 幂

```python
3 ** 10
Out[2]: 59049
2 ** -0.7
Out[3]: 0.6155722066724582
3 ** 0.5
Out[4]: 1.7320508075688772
```

### 绝对值

```python
abs(-1.7)
Out[3]: 1.7
```

### 求和

```python
sum([[1, 2], [], [1]], [])
Out[4]: [1, 2, 1]
```

### 最值

```python
max(1, 4)
Out[2]: 4
min([4, 2, 5, 6])
Out[3]: 2
```

### 四舍五入

```python
x, y = 1.555, 1.5555

f'{x:.2f} {y:.2f}'


'1.55 1.56'
```

## 装饰器

```python
def print_time(func):
    from time import time

    def wrapped(*args, **kwargs):
        t = time()
        res = func(*args, **kwargs)
        print(func.__name__, "time cost:", time() - t)
        return res

    return wrapped


@print_time
def makeE(n):
    E = [[0] * n for _ in range(n)]
    for i in range(n):
        E[i][i] = 1
    return E
```

## 常用操作

### 修改最大递归深度

```python
from sys import setrecursionlimit
setrecursionlimit(int(1e6))
```

### 无穷量

```python
a = float("inf")
-float("inf")
Out[2]: -inf
c = 3
c = min(c, a)
c
Out[3]: 3
type(c)
Out[4]: int
```

### 科学计数法

```python
int(1e5)
Out[2]: 100000
1e-4
Out[3]: 0.0001
```

### 不定长度输入

```python
while True:
    try:
        f(int(input()))
    except:
        break
```

### 2维列表合并

```python
[[1, 2]] + [[1, 2], [1]]
Out[3]: [[1, 2], [1, 2], [1]]
```

### 2维列表复制

```python
g = [l.copy() for l in graph]
```

### 字典

```python
import heapq
result = []
n, d, k = map(int, input().split())
record = {}
for i in range(n):
    ts, id = map(int, input().split())
    if id not in record:
        record[id] = []
    record[id].append(ts)
for id, ts in record.items():
    a = sorted(ts)
    s, m = 0, 10001
    l = 0
    for R in a:
        s += 1
        while s >= k:
            m = min(m, R - a[l] + 1)
            s -= 1
            l += 1
    if m <= d:
        heapq.heappush(result, id)
while result:
    print(heapq.heappop(result))
```

### 元组

```python
b = (2, 3)
a = (1,)
a + b
Out[7]: (1, 2, 3)
```

### 集合交并

```python
a = {1, 2, 3}
b = {2, 4, 6}
a | b
{1, 2, 3, 4, 6}
a & b
{2}
```

### 重定向输入

```python
import sys 
sys.stdin = open('large.txt', 'r') 
```

### 测速指令

ipython特性

```python
def f(n):
    n <<= 20
    return [i for i in range(n)]
%timeit f(1)
54.1 ms ± 1.29 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```



# 标准库

## datetime

### date类

date类包含三个参数，分别为year，month，day，返回格式为year-month-day。

date对象表示理想化日历中的日期(年、月和日), 公历1年1月1日被称为第一天，依次往后推。

其中用户创建时间的类方法如下：

```python
import datetime

print("今天的日期是:",datetime.date.today())     # 今日的日期

print("使用时间戳创建的日期：",datetime.date.fromtimestamp(1234567896))   #  使用时间戳创建日期

print("使用公历序数创建的日期：",datetime.date.fromordinal(1))    # 使用公历序数创建的日期

结果如下：

今天的日期是: 2020-08-31
使用时间戳创建的日期： 2009-02-14
使用公历序数创建的日期： 0001-01-01
```

对象的属性及方法如下：

```python
import datetime

today = datetime.date(year=2020,month=8,day=31)   #  使用参数创建日期

print('date对象的年份:', today.year)    

print('date对象的月份:', today.month)   

print('date对象的日:', today.day)  

print("date对象的struct_time结构为：",today.timetuple())

print("返回当前公历日期的序数:",today.toordinal())   #  与fromordinal函数作用相反

print("当前日期为星期(其中：周一对应0):{}".format(today.weekday()))

print("当前日期为星期(其中：周一对应1):{}".format(today.isoweekday()))

print("当前日期的年份、第几周、周几(其中返回为元组):",today.isocalendar())

print("以ISO 8601格式‘YYYY-MM-DD’返回date的字符串形式：",today.isoformat())

print("返回一个表示日期的字符串(其格式如：Mon Aug 31 00:00:00 2020):",today.ctime())

print("指定格式为：",today.strftime("%Y/%m/%d"))

print("替换后的日期为：",today.replace(2019,9,29))

结果如下：

date对象的年份: 2020
date对象的月份: 8
date对象的日: 31
date对象的struct_time结构为： time.struct_time(tm_year=2020, tm_mon=8, tm_mday=31, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=0, tm_yday=244, tm_isdst=-1)
返回当前公历日期的序数: 737668
当前日期为星期(其中：周一对应0):0
当前日期为星期(其中：周一对应1):1
当前日期的年份、第几周、周几(其中返回为元组): (2020, 36, 1)
以ISO 8601格式‘YYYY-MM-DD’返回date的字符串形式： 2020-08-31
返回一个表示日期的字符串(其格式如：Mon Aug 31 00:00:00 2020): Mon Aug 31 00:00:00 2020
指定格式为： 2020/08/31
替换后的日期为： 2019-09-29
```

其中，格式化的表达式如下：

```python
%y 两位数的年份表示（00-99）
%Y 四位数的年份表示（000-9999）
%m 月份（01-12）
%d 月内中的一天（0-31）
%H 24小时制小时数（0-23）
%I 12小时制小时数（01-12）
%M 分钟数（00=59）
%S 秒（00-59）
%a 本地简化星期名称
%A 本地完整星期名称
%b 本地简化的月份名称
%B 本地完整的月份名称
%c 本地相应的日期表示和时间表示
%j 年内的一天（001-366）
%p 本地A.M.或P.M.的等价符
%U 一年中的星期数（00-53）星期天为星期的开始
%w 星期（0-6），星期天为星期的开始
%W 一年中的星期数（00-53）星期一为星期的开始
%x 本地相应的日期表示
%X 本地相应的时间表示
%Z 当前时区的名称
%% %号本身
```

## itertools

### itertools.combinations

求列表或生成器中指定数目的元素不重复的所有组合

```python
>>> x = itertools.combinations(range(4), 3)
>>> print(list(x))
[(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
```

### itertools.permutations

产生指定数目的元素的所有排列(顺序有关)

```python
>>> x = itertools.permutations(range(4), 3)
>>> print(list(x))
[(0, 1, 2), (0, 1, 3), (0, 2, 1), (0, 2, 3), (0, 3, 1), (0, 3, 2), (1, 0, 2), (1, 0, 3), (1, 2, 0), (1, 2, 3), (1, 3, 0), (1, 3, 2), (2, 0, 1), (2, 0, 3), (2, 1, 0), (2, 1, 3), (2, 3, 0), (2, 3, 1), (3, 0, 1), (3, 0, 2), (3, 1, 0), (3, 1, 2), (3, 2, 0), (3, 2, 1)] 
```

## functools

### @functools.lru_cache()

`lru_cache()` 装饰器是 *缓存淘汰算法*(最近较少使用)的一种实现。其使用函数的参数作为key结果作为value缓存在hash结构中(因此函数的参数必须是hashable)，如果后续使用相同参数再次调用将从hash从返回结果。同时装饰器还添加了检查缓存转态方法(`cache_info()`)和清空缓存方法(`cache_clear()`)给函数。

```python
class Cache:
    def __init__(self):
        from sys import setrecursionlimit
        from functools import lru_cache

        setrecursionlimit(int(1e6))

        self.lru_cache = lru_cache

    def func(self):
        return self.lru_cache(maxsize=None)     # return key is function
```

### @functools.total_ordering

```python
@functools.total_ordering
class Node(object):
    def __init__(self, frequency, char = ""):
        self.frequency = frequency
        self.char = char
        self.left = None
        self.right = None
    def __eq__(self, other):
        return self.frequency == other.frequency
    def __gt__(self, other):
        return self.frequency > other.frequency
```

## deque

```python
from collections import deque
d = deque()
d.append(1)
d.append(2)
d.appendleft(3)
len(d)
Out[3]: 3
d.clear()
d
Out[6]: deque([])
d.extend([2, 4, 6])
d
Out[8]: deque([2, 4, 6])
d.extendleft(["a", "b", "c"])
d
Out[10]: deque(['c', 'b', 'a', 2, 4, 6])
d.pop()
Out[11]: 6
d.popleft()
Out[12]: 'c'
for i in d:
    print(i)
b
a
2
4
d[0]
Out[14]: 'b'
d[-1]
Out[15]: 4
```

## heapq

### 最小堆

将x压入堆中

```python
heappush(heap, x)
```

从堆中弹出最小的元素

```python
heappop(heap) 
```

让列表具备堆特征

```python
heapify(heap) 
```

返回iter中n个最小的元素

```python
nsmallest(n, iter)
```

## math库

### 排列组合

```
math.factorial(5)
120

math.perm(5, 3)
60

math.comb(5, 3)
10
```

### 对数

```python
math.log(2)
Out[13]: 0.6931471805599453
math.log(2, math.e)
Out[14]: 0.6931471805599453
math.log(4, 2)
Out[15]: 2.0
math.log2(4)
Out[16]: 2.0
```

### 上取整

```python
math.ceil(1.0)
Out[6]: 1
math.ceil(1.3)
Out[7]: 2
type(math.ceil(1.3))
Out[8]: int
```

### 最大公约数

```python
from math import gcd
gcd(0, 0)
Out[3]: 0
gcd(21, 14)
Out[4]: 7
gcd(0, 6)
Out[5]: 6
```

## bisect库

bisect_left等价于bound(f, a) + 1，f使小于target真

bisect等价于bound(f, a) + 1，f使小于等于target真

```python
import bisect
l = [1, 2, 4, 4, 4, 5]
bisect.bisect_left(l, 4)
Out[2]: 2
bisect.bisect_left(l, 3)
Out[3]: 2
bisect.bisect_left(l, 6)
Out[4]: 6
bisect.bisect_left(l, 1)
Out[5]: 0
bisect.bisect_left(l, 0)
Out[6]: 0
bisect.bisect(l, 4)
Out[3]: 5
bisect.bisect(l, 3)
Out[4]: 2
bisect.bisect(l, 1)
Out[5]: 1
bisect.bisect(l, 0)
Out[6]: 0
```

# 数学知识

## 数论

### 快速幂

随时取余

![img](https://s2.loli.net/2022/03/19/Clqt6g9f3kayG15.jpg)

```python
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
```

### 裴蜀定理

$(a, b)=d，ax+by=d一定有解，ax+by都是d倍数$

$(a, b)=1，存在下界，大于下界的数都可以由a，b的正整数系数线性表出$

以上结论易推广到多个数

这里给出一个不精确的下界所有互素数之积，显然大于等于所有互素数之积的数可以被素数的正整数系数线性表出
### 不定方程

#### 定理

![image-20220319090632543](https://s2.loli.net/2022/03/19/dhfyRCwQA8KJVme.png)

![image-20220319090705682](https://s2.loli.net/2022/03/19/bkuKqIUh5E1J6wT.png)

#### 通解

```python
def equation(a, b, c):
    d, x, y = exgcd(a, b)
    if c % d:
        return
    a //= d
    b //= d
    c //= d
    x *= c
    y *= c
    return lambda t: (x + b * t, y - a * t)
```

### 线性筛

```python
def linear_sieve(n):    # primes in [2, n]
    is_prime, primes = [True] * (n + 1), []

    for p in range(2, n + 1):
        if is_prime[p]:
            primes.append(p)

            for i in range(p * p, n + 1, p):
                is_prime[i] = False

    return primes
```

### 合数中最小与最大质因数的上确界

#### 引理

合数能唯一分解成质因数之积

合数质因数越少，每个质因数越大

#### 推论

仅能分解为两个质因数之积的合数的最小与最大质因数的上确界是任一合数的最小与最大质因数的上确界

质因数分布越分散，最小质因数越小，最大质因数越大，反之同理

#### 结论

##### 合数n最小质因数的上确界

$$
p_m=\sqrt{n}
$$

##### 合数n最大质因数的上确界

$$
p_M=\frac{n}{2}
$$

#### 应用

##### 判断质数n

判断质数只需判断其最小质因数是否存在
$$
判断是否存在闭区间[2,p_m]中的质数整除n
$$

##### 质数分解n

先不妨设n是合数
$$
则其全体质因数就是闭区间[2,p_M]中所有整除n的质数
$$
若其中没有数整除n，则n是质数，分解后还是n

### 判断质数

```python
def is_prime(n):
    sqrt = int(n ** 0.5)
    primes = linear_sieve(sqrt)
    
    for i in primes:
        if n % i == 0:
            return False
        
    return True
```

### 素数分解

```python
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
```

### 因数分解

```python
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
```

### 约数个数

```python
def count(n):
    factors = breakdown(n)
    
    return len(factors.values())
```

### 约数和

```python
def factors_sum(n):
    s = 1
    for p, e in breakdown(n).items():
        s *= sum(map(lambda i: p ** i, range(e + 1)))
    return s
```

### 扩展欧几里得

```python
def exgcd(a, b):
    a_neg, b_neg = False, False
    if a < 0:
        a = -a
        a_neg = True
    if b < 0:
        b = -b
        b_neg = True

    x0, y0 = 1, 0       # 本轮使A,B线性组合等于a的系数
    x1, y1 = 0, 1       # 下轮使A,B线性组合等于a的系数
    while b:
        q = a // b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
        a, b = b, a % b

    if a_neg:
        x0 = -x0
    if b_neg:
        y0 = -y0
    return a, x0, y0
```

## 组合

### 卡特兰数

所有包含n个1，n个-1的序列中前缀和始终非负且最终为0的序列数
$$
C^n_{2n}-c^{n+1}_{2n}=\frac{C^n_{2n}}{n+1}
$$

#### 进出栈序列

n 个元素进栈序列为：1，2，3，4，...，n，则有多少种出栈的元素排列

合法序列前缀和始终非负且最终为0

![扫描全能王 2022-03-24 11.01](https://s2.loli.net/2022/03/24/gECpY4h9SrunRyJ.jpg)

#### 括号序列

![image-20220324110740693](https://s2.loli.net/2022/03/24/GnCl37yEIhB1pz5.png)

```python
from math import comb
C = lambda n: comb(n << 1, n) // (n + 1)
```

## 博弈论

### Bash Game

```python
def bash(m, n):
    if n % (m + 1):
        return True
    return False
```

### Nim Game

![img](https://s2.loli.net/2022/03/22/LcXNhO5yEmgd2Ci.jpg)

```python
data = [int(x) for x in input().split()]
ans = 0
for i in data:
    ans ^= i
if ans:
    print("Yes")
else:
    print("No")
```

### 异或数列

```python
def play(data):
    ans = 0
    for i in data:
        ans ^= i
    if not ans:
        return 0
    one = [0] * 20
    def record(x):
        for i in range(20):
            if x & 1:
                one[i] += 1
            x >>= 1
    for i in data:
        record(i)
    for i in range(19, -1, -1):
        if one[i] & 1:
            zero = n - one[i]
            if one[i] == 1 or not (zero & 1):
                return 1
            return -1
for i in range(int(input())):
    l = [int(x) for x in input().split()]
    n = l[0]
    print(play(l[1:]))
```



# 图论

## 搜索

### dfs

```python
m, n = map(int, input().split())
inf = m * n
track = [[True] * m for i in range(n)]
g = [[int(x) for x in input().split()] for i in range(n)]
total = sum(map(sum, g))
half = total // 2
def solution():
    if half + half != total:
        return 0
    ans = inf
    def dfs(x, y, s, cnt):
        nonlocal ans
        s += g[x][y]
        cnt += 1
        if s > half:
            return
        if s == half:
            ans = min(ans, cnt)
            return
        def push(x, y):
            if 0 <= x < n and 0 <= y < m:
                if track[x][y]:
                    track[x][y] = False
                    dfs(x, y, s, cnt)
                    track[x][y] = True
        push(x + 1, y)
        push(x - 1, y)
        push(x, y + 1)
        push(x, y - 1)
    dfs(0, 0, 0, 0)
    if ans < inf:
        return ans
    return 0
print(solution())
```

### bfs

```python
from collections import deque
q = deque()
n, m = map(int, input().split())
graph, absent = [[] for i in range(n + 1)], [True] * (n + 1)
for i in range(m):
    a, b = map(int, input().split())
    graph[a].append(b)
def bfs():
    steps, absent[1] = -1, False
    q.append(1)
    while q:
        steps += 1
        for i in range(len(q)):
            v = q.popleft()
            if v == n:
                return steps
            for w in graph[v]:
                if absent[w]:
                    absent[w] = False
                    q.append(w)
    return -1
print(bfs())
```

#### flood fill

```python
from collections import deque
n = int(input())
graph, track = [input() for i in range(n)], [[True] * n for i in range(n)]
def bfs(x, y):
    filled, q = True, deque()
    def push(x, y):
        if 0 <= x < n and 0 <= y < n:
            if graph[x][y] == "#" and track[x][y]:
                track[x][y] = False
                q.append((x, y))
            elif graph[x][y] == ".":
                return True
    push(x, y)
    while q:
        a, b = q.popleft()
        sea = False
        if push(a + 1, b):
            sea = True
        if push(a - 1, b):
            sea = True
        if push(a, b + 1):
            sea = True
        if push(a, b - 1):
            sea = True
        if not sea:
            filled = False
    return filled
cnt = 0
for i in range(n):
    for j in range(n):
        if graph[i][j] == "#" and track[i][j]:
            if bfs(i, j):
                cnt += 1
print(cnt)
```

#### 最短路计数

```python
from collections import deque
mod = 100003
n, m = map(int, input().split())
graph = [[] for i in range(n + 1)]
for i in range(m):
    x, y = map(int, input().split())
    if x != y:
        graph[x].append(y)
        graph[y].append(x)
dist, paths = [float("inf")] * (n + 1), [0] * (n + 1)
q = deque()
dist[1], paths[1] = 0, 1
q.append(1)
while q:
    for i in range(len(q)):
        v = q.popleft()
        for vi in graph[v]:
            if dist[vi] > dist[v]:
                paths[vi] = (paths[vi] + paths[v]) % mod
                if dist[vi] == float("inf"):
                    dist[vi] = dist[v] + 1
                    q.append(vi)
for i in range(1, n + 1):
    print(paths[i])
```

## 拓扑排序

```python
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
```

## 最短路径

![](https://cdn.acwing.com/media/article/image/2019/12/13/1833_db6dffa81d-37ff39642fd8f74476ddcd99944d1b4.png)

![image-20220306213253357](https://s2.loli.net/2022/03/06/Dy4eP5hfGHl7k3d.png)

### Floyd

多源 有负权边 无负权环

```python
n, m, k = map(int, input().split())
graph = [[float("inf")] * (n + 1) for i in range(n + 1)]
for i in range(1, n + 1):
    graph[i][i] = 0
for i in range(m):
    x, y, z = map(int, input().split())
    graph[x][y] = min(graph[x][y], z)
for v in range(1, n + 1):
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            graph[i][j] = min(graph[i][j], graph[i][v] + graph[v][j])
for i in range(k):
    x, y = map(int, input().split())
    d = graph[x][y]
    if d == float("inf"):
        print("impossible")
    else:
        print(d)
```

### Dijkstra

单源 无负权边

收集距离源点最近点的过程

```python
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
```

### Bellman-Ford

有边数限制的最短路

```python
n, m, k = map(int, input().split())
edges, dist = {}, [float("inf")] * (n + 1)
for i in range(m):
    x, y, z = map(int, input().split())
    if (x, y) not in edges:
        edges[(x, y)] = z
    else:
        edges[(x, y)] = min(edges[(x, y)], z)
def dp(n, m, k):	
    dist[n] = 0
    for i in range(k):
        last_dist = dist.copy()
        for V, w in edges.items():
            v1, v2 = V
            dist[v2] = min(dist[v2], last_dist[v1] + w)
    return dist[m]
d = dp(1, n, k)
if d == float("inf"):
    print("impossible")
else:
    print(d)
```

### SPFA

单源 有负权边 有负权环

#### 最短路

```python
def spfa(n, graph, a, b):
    from collections import deque
    dist, q, not_in_q = [float("inf")] * (n + 1), deque(), [True] * (n + 1)
    def push(v):
        q.append(v)
        not_in_q[v] = False
    dist[a] = 0
    push(a)
    while q:
        v = q.popleft()
        not_in_q[v] = True
        for vi, wi in graph[v].items():
            if dist[vi] > dist[v] + wi:
                dist[vi] = dist[v] + wi
                push(vi)
    return dist[b]
```

#### 判断负环

```python
from collections import deque
n, m = map(int, input().split())
graph = [{i: 0 for i in range(1, n + 1)}] + [{} for i in range(n)]
for i in range(m):
    x, y, z = map(int, input().split())
    if y not in graph[x]:
        graph[x][y] = z
    else:
        graph[x][y] = min(graph[x][y], z)
def spfa():
    dist, q, not_in_q = [float("inf")] * (n + 1), deque(), [True] * (n + 1)
    cnt = [0] * (n + 1)
    def push(v):
        q.append(v)
        not_in_q[v] = False
        cnt[v] += 1
        if cnt[v] > n + 1:
            return True
    dist[0] = 0
    push(0)
    while q:
        v = q.popleft()
        not_in_q[v] = True
        for vi, wi in graph[v].items():
            if dist[vi] > dist[v] + wi:
                dist[vi] = dist[v] + wi
                if push(vi):
                    return True
print("Yes" if spfa() else "No")
```

## 最小生成树

### prim

```python
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
        cnt = 0

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

            cnt += dist[u]
            update(u)

        return cnt
```

### kruskal

```python
class EdgeSet:
    def __init__(self, n):
        self.n = n
        self.edges = {}

    def add(self, u, v, w):
        if u == v:
            return

        if u > v:
            u, v = v, u

        pair = (u, v)
        self.edges[pair] = min(self.edges[pair], w) if pair in self.edges else w

    def sort(self):
        return sorted([(w, pair) for pair, w in edges.edges.items()])

    def kruskal(self):
        n = self.n
        
        cnt = 0
        ufs = UFS(n)

        for w, pair in self.sort():
            if not ufs.is_connected(*pair):
                ufs.union(*pair)
                
                cnt += w

        if ufs.get_size(1) < n:
            return

        return cnt
```

## 最短Hamilton路径

```python
n = int(input())
ST = 1 << n
graph = [[int(x) for x in input().split()] for i in range(n)]
dp = [[float("inf")] * n for i in range(ST)]    # 从0经过i集合到j点最短距离
dp[1][0] = 0
for st in range(ST):
    in_st = lambda v: st >> v & 1
    if in_st(0):
        for i in range(n):
            if in_st(i):
                for j in range(n):
                    if in_st(j):
                        dp[st][i] = min(dp[st][i],
                                        dp[st - (1 << i)][j] + graph[j][i])
print(dp[ST - 1][n - 1])
```



# 计算几何

```python
lines = set()
n, m = map(int, input().split())
for y in range(n):
    for x in range(m):
        for yi in range(y + 1, n):
            for xi in range(m):
                if x != xi:
                    k = (y - yi) / (x - xi)
                    b = y - k * x
                    lines.add((round(k, 8), round(b, 8)))
print(n + m + len(lines))
```

## 平面几何

### 偏差量

```python
eps = 1e-8
```

### 等于0

```python
def zero(x):
    if x >= eps:
        return 1
    if x <= -eps:
        return -1
    return 0
```

### 相等

```python
def eq(x, y):
    if abs(x - y) < eps:
        return 0
    if x > y:
        return 1
    return -1
```

### 平面划分

```python
eps = 1e-8
def eq(x, y):
    if abs(x - y) < eps:
        return 0
    if x > y:
        return 1
    return -1
n = int(input())
lines = {tuple(int(x) for x in input().split()) for i in range(n)}
track, cnt = [lines.pop()], 2
for a1, b1 in lines:
    cnt += 1
    points = set()
    for a2, b2 in track:
        if eq(a1, a2):
            p = (-(b1 - b2) / (a1 - a2), (a1 * b2 - a2 * b1) / (a1 - a2))
            if p not in points:
                cnt += 1
                points.add(p)
    track.append((a1, b1))
print(cnt)
```

## 点叉积

```python
dot = lambda a, b: sum(map(lambda i: a[i] * b[i], range(len(a))))
size = lambda a: dot(a, a) ** 0.5
def cross(a, b):
    x1, y1 = a
    x2, y2 = b
    return x1 * y2 - x2 * y1
```

```python
def solution():
    x0, y0 = map(int, input().split())
    x1, y1 = map(int, input().split())
    x2, y2 = map(int, input().split())
    a, b = (x1 - x0, y1 - y0), (x2 - x0, y2 - y0)
    return abs(cross(a, b)) / 2
for i in range(int(input())):
    print(format(solution(), ".2f"))
```

### **点和直线的位置关系**

```python
for i in range(int(input())):
    x1, y1 = map(int, input().split())
    x2, y2 = map(int, input().split())
    x, y = map(int, input().split())
    ab = (x2 - x1, y2 - y1)
    p = (x - x1, y - y1)
    d = zero(cross(p, ab))
    if d == 0:
        print("IN")
    elif d > 0:
        print("R")
    else:
        print("L")
```

### 点和线段关系

```python
for i in range(int(input())):
    x1, y1 = map(int, input().split())
    x2, y2 = map(int, input().split())
    x, y = map(int, input().split())
    ab = (x2 - x1, y2 - y1)
    p = (x - x1, y - y1)
    d, product = zero(cross(p, ab)), zero(dot(p, ab))
    if d == 0 and product >= 0:
        print("Yes")
    else:
        print("No")
```

### **点到直线的距离**

```python
def height(a, b, p):
    x1, y1 = a
    x2, y2 = b
    x, y = p
    v = (x - x1, y - y1)
    ab = (x2 - x1, y2 - y1)
    s = abs(cross(v, ab))
    return s / size(ab)
```

### **点关于直线的对称点**

```python
def plus(a, b):
    x1, y1 = a
    x2, y2 = b
    return (x1 + x2, y1 + y2)
def minus(a, b):
    x1, y1 = a
    x2, y2 = b
    return (x1 - x2, y1 - y2)
def K(p, k):
    x, y = p
    return (k * x, k * y)
def dot(a, b):
    x1, y1 = a
    x2, y2 = b
    return x1 * x2 + y1 * y2
def projection(a, b, p):
    v = minus(b, a)
    l = minus(p, a)
    k = dot(v, l) / dot(v, v)
    return plus(a, K(v, k))
def symmetry(a, b, p):
    return minus(K(projection(a, b, p), 2), p)
print(symmetry((0, 0), (1, 1), (0, 3)))
```

### **两条直线的位置关系**

```python
def relation(a, b, c, d):
    v1 = minus(b, a)
    v2 = minus(d, c)
    if zero(cross(v1, v2)) == 0:
        v = minus(c, a)
        if zero(cross(v1, v)) == 0:
            return "重合"
        else:
            return "平行"
    else:
        return "相交"
```

## 扫描线（无离散化）

```python
length = 10001
L = length << 2 | 1
left, right, cnt, s = [0] * L, [0] * L, [0] * L, [0] * L
ls, rs = lambda k: k << 1, lambda k: k << 1 | 1
m = lambda k: left[k] + right[k] >> 1
def build(k, l, r):
    left[k], right[k] = l, r
    if l < r:
        build(ls(k), l, m(k))
        build(rs(k), m(k) + 1, r)
build(1, 1, length)
def modify(k, l, r, x):
    if l <= left[k] and right[k] <= r:
        cnt[k] += x
    else:
        if l <= m(k):
            modify(ls(k), l, r, x)
        if m(k) < r:
            modify(rs(k), l, r, x)
    if cnt[k] > 0:
        s[k] = right[k] - left[k] + 1
    elif left[k] == right[k]:
        s[k] = 0
    else:
        s[k] = s[ls(k)] + s[rs(k)]
n = int(input())
lines = []
for i in range(n):
    x1, y1, x2, y2 = map(int, input().split())
    lines.append((x1, y1, y2, 1))
    lines.append((x2, y1, y2, -1))
lines.sort()
result = 0
for i in range(n << 1):
    if i:
        result += (lines[i][0] - lines[i - 1][0]) * s[1]
    modify(1, lines[i][1] + 1, lines[i][2], lines[i][3])
print(result)
```



# 位运算

## 右移左移

```python
13 >> 1
Out[3]: 6
1 << 3
Out[4]: 8

```

## 取位

```python
n = 13    # 1101
for i in range(4):
    print((n >> i) & 1)
    
1
0
1
1
```

## 置1

```python
def setbit(x, k):		# 从第0位计数
    n = 1 << k
    return x | n
setbit(8, 1)
Out[2]: 10
```

## 异或

逐位异真，同假

不进位加法

遇0不变，遇1反转

```python
a = 10010 # 18
b = 11100 # 
a = 18 # 10010
b = 28 # 11100
c = a ^ b
c	# 01110
Out[10]: 14
True ^ True
Out[12]: False
True ^ False
Out[13]: True
```

### 只出现一次的数字

数组除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素

```python
def singleNumber(nums):
    ans = 0

    for i in nums:
        ans ^= i

        return ans
```

## lowbit

函数的值是x的二进制表达式中最低位的1所对应的值

```python
lowbit = lambda x: x & (-x)
lowbit(8)
Out[3]: 8
lowbit(9)
Out[4]: 1
```



# 数据结构

## 串

### 基本操作

#### find

```python
s.find(sub, start=0, end=len(s))
```

s.find(sub, start=0, end=len(s))

matching index range: [start, end)

return the first index of the first match, or -1 if no match

#### count

```python
s.count(sub, start=0, end=len(s))
```

count the number of the whole, independent matches

### 前后缀函数

#### 最大公共真前后缀

```python
def max_fix(s):
    max_same = [0] * len(s)
    for i in range(1, len(s)):
        j = i - 1

        while j >= 0:
            pi = max_same[j]

            if s[i] == s[pi]:
                max_same[i] = pi + 1
                break

            j = pi - 1

    return max_same
```

#### 最小公共真前后缀

```python
def min_fix(s):
    min_same = max_fix(s)
    
    for i in range(1, len(s)):
        j = i   # initial j as i > 0

        while pi := min_same[j]:    # ensure j and pi > 0
            min_same[i] = pi
            j = pi - 1  # worst j is 0

    return min_same
```

#### kmp

```python
def kmp(s, p):
    new = p + '#' + s
    max_same = max_fix(new)
    s, p = len(s), len(p)

    return [i for i in range(s - p + 1) if max_same[i + 2 * p] == p]
```

## 单调栈

```python
def largestRectangleArea(heights):
    n = len(heights)
    heights = [-2] + heights + [-1]
    stack = [(-2, 0)]
    left, right = [0] * (n + 2), [0] * (n + 2)
    def push(h, p):
        top_h, top_p = stack[-1]
        while top_h >= h:
            if top_h > h:
                right[top_p] = p
            stack.pop()
            top_h, top_p = stack[-1]
        left[p] = top_p
        stack.append((h, p))
    for i in range(1, n + 2):
        push(heights[i], i)
    m = 0
    for i in range(1, n + 1):
        m = max(m, heights[i] * (right[i] - left[i] - 1))
    return m
```

## 单调队列

```python
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


class Solution:
    def maxSlidingWindow(self, nums, k):
        maxs = []

        decrease_queue = MonotonicQueue('decrease', nums[: k])
        maxs.append(decrease_queue.head())

        for old_left in range(len(nums) - k):
            new_right = old_left + k
            
            old_left = nums[old_left]
            new_right = nums[new_right]

            decrease_queue.pop(old_left)
            decrease_queue.push(new_right)

            maxs.append(decrease_queue.head())
        
        return maxs
```

## 并查集

```python
class UFS:
    def __init__(self, n):
        self.n = n
        self.f = list(range(n + 1))

        self.size = [1] * (n + 1)
        self.dist = [0] * (n + 1)

    def is_root(self, a):
        return a == self.f[a]

    def is_connected(self, a, b):
        return self.find(a) == self.find(b)

    def find(self, a):
        if not self.is_root(a):
            f = self.f[a]
            self.f[a] = self.find(f)

            pass                                # update dist

        return self.f[a]

    def union(self, a, b):                      # a branch, b root
        root_a, root_b = self.find(a), self.find(b)

        if root_a != root_b:
            self.f[root_a] = root_b

            pass                                # update dist
            self.size[root_b] += self.size[root_a]

    def get_roots(self):
        return [i for i in range(1, self.n + 1) if self.is_root(i)]

    def get_size(self, a):
        root = self.find(a)
        return self.size[root]

    def calc_dist(self, a, b):
        if not self.is_connected(a, b):
            return -1

        return self.dist[a] - self.dist[b]
```

## 前缀和

### 静态

#### 1维

```python
n, m = map(int, input().split())
l = [0] + [int(x) for x in input().split()]
s = [0] * (n + 1)
for i in range(1, n + 1):
    s[i] = s[i - 1] + l[i]
def query():
    l, r = [int(x) for x in input().split()]
    return s[r] - s[l - 1]
for i in range(m):
    print(query())
```

#### 2维

```python
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
```

### 树状数组

```python
n, times = map(int, input().split())
lowbit = list(map(lambda x: x & (-x), range(n + 1)))
l = [0] + [int(x) for x in input().split()]
s = [0] * (n + 1)
def add(k, x):
    l[k] += x
    pos = k
    while pos < n + 1:
        s[pos] += x
        pos += lowbit[pos]
def query(k):
    result = 0
    pos = k
    while pos:
        result += s[pos]
        pos -= lowbit[pos]
    return result
for i in range(1, n + 1):
    add(i, l[i])
for i in range(times):
    k, a, b = [int(x) for x in input().split()]
    if k:
        add(a, b)
    else:
        print(query(b) - query(a - 1))
```

## 差分

### 1维

```python
n, m = map(int, input().split())
l = [0] + [int(x) for x in input().split()] + [0]
delt = [0] * (n + 2)
for i in range(1, n + 1):
    delt[i] = l[i] - l[i - 1]
def modify(l, r, x):
    delt[l] += x
    delt[r + 1] -= x
for i in range(m):
    a, b, c = map(int, input().split())
    modify(a, b, c)
for i in range(1, n + 1):
    l[i] = l[i - 1] + delt[i]
    print(l[i], end = " ")
```

### 2维

```python
n, m, q = map(int, input().split())
delt = [[0] * (m + 2) for i in range(n + 2)]
def modify(x1, y1, x2, y2, c):
    delt[x1][y1] += c
    delt[x2 + 1][y2 + 1] += c
    delt[x2 + 1][y1] -= c
    delt[x1][y2 + 1] -= c
for i in range(1, n + 1):
    data = map(int, input().split())
    for j in range(1, m + 1):
        modify(i, j, i, j, next(data))
for i in range(q):
    a, b, c, d, e = map(int, input().split())
    modify(a, b, c, d, e)
matrix = [[0] * (m + 2) for i in range(n + 2)]
for i in range(1, n + 1):
    for j in range(1, m + 1):
        matrix[i][j] = delt[i][j] - matrix[i - 1][j - 1] + \
                       matrix[i][j - 1] + matrix[i - 1][j]
        print(matrix[i][j], end = " ")
    print()
```

## st表

```python
from math import log2
def process(n, l):
    bottom = int(log2(n))
    st = [l] + [[0] * n for i in range(bottom)]
    for i in range(1, bottom + 1):
        j = 0
        tail = j + (1 << i) - 1
        while tail <= n - 1:
            interval = 1 << (i - 1)
            st[i][j] = max(st[i - 1][j], st[i - 1][j + interval])
            j += 1
            tail = j + (1 << i) - 1
    return st
n, m = map(int, input().split())
st_table = process(n, [int(x) for x in input().split()])
def query(l, r):
    k = int(log2(r - l + 1))
    interval = (1 << k) - 1
    return max(st_table[k][l], st_table[k][r - interval])

for i in range(m):
    a, b = [int(x) - 1 for x in input().split()]
    print(query(a, b))
```

## 平衡树

### 线段树

记录n个元素，线段树空间开到4*n

#### 单点修改

```python
length, times = map(int, input().split())
L = length << 2 | 1
left, right, s = [0] * L, [0] * L, [0] * L
data = [int(x) for x in input().split()]
def push(k):
    s[k] = max(s[k << 1], s[(k << 1) + 1])
def build(k, l, r):
    left[k], right[k] = l, r
    if l == r:
        s[k] = data[l - 1]
    else:
        m = l + r >> 1
        build(k << 1, l, m)
        build((k << 1) + 1, m + 1, r)
        push(k)
build(1, 1, length)
def query(k, l, r):
    if l <= left[k] and right[k] <= r:
        return s[k]
    m = left[k] + right[k] >> 1
    result = 0
    if l <= m:
        result = query(k << 1, l, r)
    if m < r:
        result = max(result, query((k << 1) + 1, l, r))
    return result
def modify(k, n, x):
    if left[k] == right[k]:
        s[k] = x
    else:
        m = left[k] + right[k] >> 1
        if n <= m:
            modify(k << 1, n, x)
        else:
            modify((k << 1) + 1, n, x)
        push(k)
for i in range(times):
    k, a, b = map(int, input().split())
    if k:
        modify(1, a, b)
    else:
        print(query(1, a, b))
```

#### 区间修改

##### 求和

```python
ls = lambda k: 2 * k
rs = lambda k: 2 * k + 1


class SumSegmentTree:
    def __init__(self, array):  # element index starts from 0
        self.n = len(array)
        size = 4 * self.n

        self.array = [0] + array

        self.left, self.right = [0] * size, [0] * size
        self.value, self.tag = [0] * size, [0] * size

        self.build(1, 1, self.n)

    def m(self, k):
        return (self.left[k] + self.right[k]) // 2

    def length(self, k):
        return self.right[k] - self.left[k] + 1

    def update(self, k, delt):
        self.tag[k] += delt
        self.value[k] += delt * self.length(k)

    def push_up(self, k):
        self.value[k] = self.value[ls(k)] + self.value[rs(k)]

    def push_down(self, k):
        delt = self.tag[k]
        self.update(ls(k), delt), self.update(rs(k), delt)
        self.tag[k] = 0

    def build(self, k, l, r):
        self.left[k], self.right[k] = l, r

        if l == r:
            self.value[k] = self.array[l]
        else:
            m = self.m(k)

            self.build(ls(k), l, m)
            self.build(rs(k), m + 1, r)
            self.push_up(k)

    def _modify(self, k, a, b, delt):
        if a <= self.left[k] and self.right[k] <= b:
            self.update(k, delt)
        else:
            self.push_down(k)
            m = self.m(k)

            if a <= m:
                self._modify(ls(k), a, b, delt)
            if b > m:
                self._modify(rs(k), a, b, delt)
            self.push_up(k)

    def _query(self, k, a, b):
        if a <= self.left[k] and self.right[k] <= b:
            return self.value[k]

        self.push_down(k)

        res = 0

        m = self.m(k)
        if a <= m:
            res += self._query(ls(k), a, b)
        if m < b:
            res += self._query(rs(k), a, b)

        return res

    def modify(self, a, b, delt):
        self._modify(1, a, b, delt)

    def query(self, a, b):
        return self._query(1, a, b)
```

###### 链式

```python
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
```

##### 最值

```python
ls = lambda k: 2 * k
rs = lambda k: 2 * k + 1


class ValueSegmentTree:
    def __init__(self, array):  # element index starts from 0
        self.n = len(array)
        size = 4 * self.n

        self.array = [0] + array

        self.left, self.right = [0] * size, [0] * size

        self.max = [0] * size
        self.min = [0] * size

        self.build(1, 1, self.n)

    def m(self, k):
        return (self.left[k] + self.right[k]) // 2

    def push_up(self, k):
        self.max[k] = max(self.max[ls(k)], self.max[rs(k)])
        self.min[k] = min(self.min[ls(k)], self.min[rs(k)])

    def build(self, k, l, r):
        self.left[k], self.right[k] = l, r

        if l == r:
            self.max[k] = self.array[l]
            self.min[k] = self.array[l]
        else:
            m = self.m(k)

            self.build(ls(k), l, m)
            self.build(rs(k), m + 1, r)
            self.push_up(k)

    def _query(self, k, a, b):
        if a <= self.left[k] and self.right[k] <= b:
            return self.max[k], self.min[k]

        M, m = 0, float('inf')

        if a <= self.m(k):
            _M, _m = self._query(ls(k), a, b)
            M = max(M, _M)
            m = min(m, _m)
        if self.m(k) < b:
            _M, _m = self._query(rs(k), a, b)
            M = max(M, _M)
            m = min(m, _m)

        return M, m

    def query(self, a, b):
        return self._query(1, a, b)
```

##### 公约数

```python
class GCDSegmentTree:
    def __init__(self, array):  # element index starts from 0
        from math import gcd

        self.gcd = gcd

        self.n = len(array)
        size = 4 * self.n

        self.array = [0] + array

        self.left, self.right = [0] * size, [0] * size
        self.value, self.tag = [0] * size, [0] * size

        self.build(1, 1, self.n)

    def m(self, k):
        return (self.left[k] + self.right[k]) // 2

    def length(self, k):
        return self.right[k] - self.left[k] + 1

    def push_up(self, k):
        self.value[k] = self.gcd(self.value[ls(k)], self.value[rs(k)])

    def build(self, k, l, r):
        self.left[k], self.right[k] = l, r

        if l == r:
            self.value[k] = self.array[l]
        else:
            m = self.m(k)

            self.build(ls(k), l, m)
            self.build(rs(k), m + 1, r)
            self.push_up(k)

    def _query(self, k, a, b):
        if a <= self.left[k] and self.right[k] <= b:
            return self.value[k]

        res = 0

        m = self.m(k)
        if a <= m:
            res = self.gcd(res, self._query(ls(k), a, b))
        if m < b:
            res = self.gcd(res, self._query(rs(k), a, b))

        return res

    def query(self, a, b):
        return self._query(1, a, b)
```

##### 区间加，区间乘，查询区间和

```python
class ModCalculation:
    def __init__(self, mod=0):
        self.mod = mod

    def get_mod(self, a):
        return a % self.mod if self.mod else a

    def add(self, a, b):
        return (a + b) % self.mod if self.mod else a + b

    def sum(self, iterable):
        s = 0
        for i in iterable:
            s = self.add(s, i)
        return s

    def mul(self, a, b):
        return (a * b) % self.mod if self.mod else a * b

    def power(self, a, n):
        res = 1

        while n:
            res = self.mul(res, a)

        return res

    def quick_power(self, a, n):
        res = 1

        while n:
            if n & 1:
                res = self.mul(res, a)

            a = self.mul(a, a)
            n >>= 1

        return res


ring = ModCalculation(571373)

ls = lambda k: 2 * k
rs = lambda k: 2 * k + 1


class SegmentTree:
    def __init__(self, array):  # element index starts from 0
        self.n = len(array)
        self.size = 4 * self.n

        self.array = [0] + array

        self.left, self.right = [0] * self.size, [0] * self.size
        self.value = [0] * self.size

        self.add_tag = [0] * self.size
        self.mul_tag = [1] * self.size

        self.build(1, 1, self.n)

    def m(self, k):
        return (self.left[k] + self.right[k]) // 2

    def length(self, k):
        return self.right[k] - self.left[k] + 1

    def set_update(self, cmd):
        self.update = self.update_add if cmd == 2 else self.update_mul

    # def update(self, k, delt):
    #     # self.tag[k] += delt
    #     # self.value[k] += delt * self.length(k)
    #     pass
    
    def update_add(self, k, delt):
        self.add_tag[k] = ring.add(self.add_tag[k], delt)
        self.value[k] = ring.add(self.value[k], ring.mul(self.length(k), delt))
        
    def update_mul(self, k, delt):
        self.add_tag[k] = ring.mul(self.add_tag[k], delt)
        self.mul_tag[k] = ring.mul(self.mul_tag[k], delt)
        self.value[k] = ring.mul(self.value[k], delt)

    def push_up(self, k):
        self.value[k] = ring.add(self.value[ls(k)], self.value[rs(k)])

    def push_down(self, k):
        add_delt = self.add_tag[k]
        mul_delt = self.mul_tag[k]

        self.update_mul(ls(k), mul_delt), self.update_mul(rs(k), mul_delt)
        self.update_add(ls(k), add_delt), self.update_add(rs(k), add_delt)

        self.add_tag[k] = 0
        self.mul_tag[k] = 1

    def build(self, k, l, r):
        self.left[k], self.right[k] = l, r

        if l == r:
            self.value[k] = self.array[l]
        else:
            m = self.m(k)

            self.build(ls(k), l, m)
            self.build(rs(k), m + 1, r)
            self.push_up(k)

    def _modify(self, k, a, b, delt):
        if a <= self.left[k] and self.right[k] <= b:
            self.update(k, delt)
        else:
            self.push_down(k)
            m = self.m(k)

            if a <= m:
                self._modify(ls(k), a, b, delt)
            if b > m:
                self._modify(rs(k), a, b, delt)
            self.push_up(k)

    def _query(self, k, a, b):
        res = 0

        if a <= self.left[k] and self.right[k] <= b:
            return self.value[k]

        self.push_down(k)
        m = self.m(k)

        if a <= m:
            res = ring.add(res, self._query(ls(k), a, b))
        if m < b:
            res = ring.add(res, self._query(rs(k), a, b))

        return res

    def modify(self, a, b, delt):
        self._modify(1, a, b, delt)

    def query(self, a, b):
        return self._query(1, a, b)


if __name__ == '__main__':
    n, m, _ = map(int, input().split())
    array = [int(x) for x in input().split()]

    tree = SegmentTree(array)

    for _ in range(m):
        iter = map(int, input().split())
        cmd, a, b = next(iter), next(iter), next(iter)

        if cmd == 3:
            print(tree.query(a, b))
        else:
            tree.set_update(cmd)
            k = next(iter)

            tree.modify(a, b, k)
```

### 树堆

#### 单点修改

##### 结构

链式结构，每个节点有权值value和随机分配的键值key。满足按权构成二叉搜索树同时按键构成最小堆的树堆存在且唯一，其效率期望符合平衡树

为了按秩访问节点，根据treap性质为节点增加size属性，同时定义push_up()方法在更新树后更新size属性

```python
get_size = lambda p: p.size if p else 0
class node:
    from random import random
    def __init__(self, value):
        self.value = value
        self.key = node.random()
        self.left, self.right = None, None
        self.size = 1
    def push_up(self):
        self.size = 1 + get_size(self.left) + get_size(self.right)
```

##### 基本操作

分裂与合并的组合可以实现大部分树的操作

###### 按值分裂

将一棵treap分裂为一棵全部值小于等于目标值的treap和一棵全部值大于目标值的treap。

对于非空节点p，当p.value <= x时，p及p的左子树全部值小于等于x，此时将p的右子树分裂为全部值小于等于目标值的l和全部值大于目标值的r，r就是全部值大于x的treap，因为p是treap，所以树堆l根节点的值与键都大于p，将l插在p的右子树后整体依然构成treap。p.value > x时同理。在递归过程中每次更新p的左右子树后更新其size从而保证size值正确

```python
def split(p, x):
    if not p:
        return None, None
    if p.value <= x:
        l, r = split(p.right, x)
        l, p.right = p, l
    else:
        l, r = split(p.left, x)
        r, p.left = p, r
    p.push_up()
    return l, r
```

###### 合并

将非空树堆l，r(max(l) <= min(r))合并

当l.key <= r.key时，将l右子树与r合并，新树堆的值与键都大于l，故将其插入l右子树出l仍为treap，l.key > r.key时同理。在递归过程中每次更新p的左右子树后更新其size从而保证size值正确

```python
def merge(l, r):
    if not l or not r:
        return l or r
    if l.key <= r.key:
        l.right, p = merge(l.right, r), l
    else:
        r.left, p = merge(l, r.left), r
    p.push_up()
    return p
```

##### 功能架构

```python
class treap:
    def __init__(self):
        self.root = None
    def insert(self, x):
        l, r = split(self.root, x - 1)
        l = merge(l, node(x))
        self.root = merge(l, r)
    def remove(self, x):
        l, r = split(self.root, x - 1)
        l1, r1 = split(r, x)
        l1 = merge(l1.left, l1.right)
        r = merge(l1, r1)
        self.root = merge(l, r)
    def rank(self, x):
        l, r = split(self.root, x - 1)
        size = get_size(l)
        self.root = merge(l, r)
        return size + 1
    def value(self, k):
        p = self.root
        while p:
            rk = get_size(p.left) + 1
            if k < rk:
                p = p.left
            elif k > rk:
                k -= rk
                p = p.right
            else:
                return p.value
    def lower(self, x):
        p, M = self.root, -float("inf")
        while p:
            if p.value < x:
                M = max(M, p.value)
                p = p.right
            else:
                p = p.left
        return M
    def upper(self, x):
        p, m = self.root, float("inf")
        while p:
            if p.value > x:
                m = min(m, p.value)
                p = p.left
            else:
                p = p.right
        return m
```

#### 区间翻转

```python
from random import random
get_size = lambda p: p.size if p else 0

class node:
    def __init__(self, x):
        self.value = x
        self.key = random()
        self.left, self.right = None, None
        self.size = 1
        self.reversed = False
    def push_up(self):
        self.size = 1 + get_size(self.left) + get_size(self.right)
    def push_down(self):
        if not self.reversed:
            return
        if self.left:
            self.left.reversed ^= True
        if self.right:
            self.right.reversed ^= True
        self.reversed = False
        self.left, self.right = self.right, self.left

def split(p, k):
    if not p:
        return None, None
    p.push_down()
    rk = 1 + get_size(p.left)
    if rk <= k:
        l, r = split(p.right, k - rk)
        l, p.right = p, l
    else:
        l, r = split(p.left, k)
        r, p.left = p, r
    p.push_up()
    return l, r
def merge(l, r):
    if not l or not r:
        return l or r
    l.push_down()
    r.push_down()
    if l.key <= r.key:
        p, l.right = l, merge(l.right, r)
    else:
        p, r.left = r, merge(l, r.left)
    p.push_up()
    return p

class treap:
    def __init__(self):
        self.root = None
    def insert(self, x):
        self.root = merge(self.root, node(x))
    def reverse(self, a, b):
        l, m = split(self.root, a - 1)
        m, r = split(m, b - a + 1)
        m.reversed ^= True
        r = merge(m, r)
        self.root = merge(l, r)
    def list(self):
        l = []
        def dfs(p):
            p.push_down()
            if p.left:
                dfs(p.left)
            l.append(p.value)
            if p.right:
                dfs(p.right)
        dfs(self.root)
        return l
```

#### 区间查改

```python
from random import random
get_size = lambda p: p.size if p else 0
get_sum = lambda p: p.sum if p else 0

class node:
    def __init__(self, value):
        self.sum = self.value = value
        self.key = random()
        self.size = 1
        self.left = self.right = None
        self.tag = 0
    def push_up(self):
        self.sum = self.value + get_sum(self.left) + get_sum(self.right)
        self.size = 1 + get_size(self.left) + get_size(self.right)
    def push_down(self):
        cover(self.left, self.tag)
        cover(self.right, self.tag)
        self.tag = 0

def cover(p, tag):
    if not p:
        return
    p.tag += tag
    p.value += tag
    p.sum += tag * get_size(p)
def split(p, k):
    if not p:
        return None, None
    p.push_down()
    rk = 1 + get_size(p.left)
    if rk <= k:
        l, r = split(p.right, k - rk)
        l, p.right = p, l
    else:
        l, r = split(p.left, k)
        r, p.left = p, r
    p.push_up()
    return l, r
def merge(l, r):
    if not l or not r:
        return l or r
    l.push_down()
    r.push_down()
    if l.key <= r.key:
        p, l.right = l, merge(l.right, r)
    else:
        p, r.left = r, merge(l, r.left)
    p.push_up()
    return p

class treap:
    def __init__(self):
        self.root = None
    def insert(self, value):
        self.root = merge(self.root, node(value))
    def modify(self, a, b, tag):
        l, m = split(self.root, a - 1)
        m, r = split(m, b - a + 1)
        cover(m, tag)
        r = merge(m, r)
        self.root = merge(l, r)
    def query(self, a, b):
        l, m = split(self.root, a - 1)
        m, r = split(m, b - a + 1)
        s = m.sum
        r = merge(m, r)
        self.root = merge(l, r)
        return s
```



# 算法

## 二分查找

对于一个分隔为两部分的列表不妨设前一部分为真，找到最后一个使题设为真的下标

```python
def bound(true, l, r):      # last number makes true in [l, r)
    if r - l == 0 or not true(l):
        return -1           # no number makes true
    
    while r - l > 1:
        m = (l + r) // 2

        if true(m):
            l = m
        else:
            r = m

    return l
```

## 滑动窗口

区间数据满足二分属性，不妨设满足题意部分为真

### 最长

连续数据增长趋近为假，求条件为真最长连续子区间长度

```python
class Window:
    from collections import deque

    def __init__(self):
        self.queue = self.deque()

    def __len__(self):
        return len(self.queue)

    def push(self, x):
        self.queue.append(x)
        
    def pop(self):
        self.queue.popleft()
        
    def true(self):
        q_set = set(self.queue)

        return len(self) == len(q_set)


class Solution:
    def lengthOfLongestSubstring(self, s):
        window = Window()

        M = 0

        for i in s:
            window.push(i)

            while not window.true():
                window.pop()

            M = max(M, len(window))

        return M
```

### 最短

连续数据增长趋近为真，求条件为真最短连续子区间长度

```python
class Window:
    from collections import deque

    def __init__(self):
        self.queue = self.deque()

        self.sum = 0

    def __len__(self):
        return len(self.queue)

    def push(self, x):
        self.queue.append(x)

        self.sum += x

    def pop(self):
        self.sum -= self.queue.popleft()

    def true(self, target):
        return self.sum >= target


class Solution:
    def minSubArrayLen(self, target, nums):
        window = Window()

        m = float('inf')

        for i in nums:
            window.push(i)
            
            while window.true(target):
                m = min(m, len(window))
                
                window.pop()

        return m if m < float('inf') else 0
```

### 计数

```python
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
```

```python
class PrefixArea:
    def __init__(self, matrix):  # matrix extend 0
        n = len(matrix)
        m = len(matrix[0])

        self.area = [[0] * m for _ in range(n)]

        for i in range(1, n):
            for j in range(1, m):
                self.area[i][j] = matrix[i][j] + self.two_area(i - 1, j - 1, i, j)

    def two_area(self, x1, y1, x2, y2):  # x1 <= x2, y1 <= y2
        return self.area[x1][y2] + self.area[x2][y1] - self.area[x1][y1]

    def query(self, x1, y1, x2, y2):
        return self.area[x2][y2] - self.two_area(x1 - 1, y1 - 1, x2, y2)


class WindowCounter:
    def __init__(self, left, matrix, up, down):
        self.left = left
        self.right = left - 1

        self.matrix = matrix

        self.up = up
        self.down = down

    def __len__(self):
        return self.right - self.left + 1

    def push(self):
        self.right += 1

    def pop(self):
        self.left += 1

    def true(self, prefix_area, limit):
        sum = prefix_area.query(self.up, self.left, self.down, self.right)
        return sum <= limit


def extend0_matrix_input(n, m):
    return [[0] * (m + 1)] + [[0] + [int(x) for x in input().split()] for _ in range(n)]


if __name__ == '__main__':
    n, m, limit = map(int, input().split())
    matrix = extend0_matrix_input(n, m)

    prefix_area = PrefixArea(matrix)
    cnt = 0

    for up in range(1, n + 1):
        for down in range(up, n + 1):  # left initial
            window_counter = WindowCounter(1, matrix, up, down)

            for _ in range(m):
                window_counter.push()

                while not window_counter.true(prefix_area, limit):
                    window_counter.pop()

                cnt += len(window_counter)

    print(cnt)
```

## dp

### 线性

#### 最长公共子序列

```python
a, b = map(int, input().split())
A, B = " " + input(), " " + input()
dp = [[0] * (b + 1) for i in range(a + 1)]      # A from 1 to i B from 1 to j sub max
for i in range(1, a + 1):
    for j in range(1, b + 1):
        if A[i] == B[j]:
            dp[i][j] = dp[i - 1][j - 1] + 1
        else:
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
print(dp[a][b])
```

#### LIS

##### dp

```python
def LIS(arr):
    n = len(arr)
    arr.insert(0, -float("inf"))
    dp = [0] * (n + 1)
    M = 0
    for i in range(1, n + 1):
        for j in range(i):
            if arr[j] < arr[i]:
                dp[i] = max(dp[i], dp[j] + 1)
                M = max(M, dp[i])
    ans = []
    for i in range(n, 0, -1):
        if dp[i] == M:
            ans.append(arr[i])
            M -= 1
        if M == 0:
            break
    return ans[-1: : -1]
```

##### 二分

```python
def LIS(arr):
    n, w = len(arr), [-float("inf")] + arr
    min_tail, size = [-float("inf")] + [float("inf")] * n, [0] * (n + 1)
    find = lambda x: bound(lambda i: min_tail[i] < x, range(n + 1))
    for i in range(1, n + 1):
        j = find(w[i])
        size[i] = j + 1
        min_tail[j + 1] = min(min_tail[j + 1], w[i])
    M, ans = max(size), []
    for i in range(n, 0, -1):
        if size[i] == M:
            ans.append(w[i])
            M -= 1
        if M == 0:
            break
    return ans[n: : -1]
```

### 背包

#### 01

##### 递归

```python
from functools import lru_cache
n, k = map(int, input().split())
v, w = [0], [0]
for i in range(n):
    vi, wi = map(int, input().split())
    v.append(vi)
    w.append(wi)
@lru_cache()
def max_w(n, k):
    if n == 0 or k == 0:
        return 0
    M = max_w(n - 1, k)
    if k >= v[n]:
        M = max(M, 
                max_w(n - 1, k - v[n]) + w[n])
    return M
print(max_w(n, k))
```

##### dp

```python
n, k = map(int, input().split())
v, w = [0], [0]
for i in range(n):
    vi, wi = map(int, input().split())
    v.append(vi)
    w.append(wi)
dp = [[0] * (k + 1) for i in range(n + 1)]
for i in range(1, n + 1):
    for j in range(1, k + 1):
        dp[i][j] = dp[i - 1][j]
        if j >= v[i]:
            dp[i][j] = max(dp[i][j],
                           dp[i - 1][j - v[i]] + w[i])
print(dp[n][k])
```

#### 完全

##### 递归

```python
from functools import lru_cache
n, k = map(int, input().split())
v, w = [0], [0]
for i in range(n):
    vi, wi = map(int, input().split())
    v.append(vi)
    w.append(wi)
@lru_cache()
def max_w(n, k):
    if n == 0 or k == 0:
        return 0
    M = max_w(n - 1, k)
    if k >= v[n]:
        M = max(M,
                max_w(n, k - v[n]) + w[n])
    return M
print(max_w(n, k))
```

##### dp

```python
n, k = map(int, input().split())
v, w = [0], [0]
for i in range(n):
    vi, wi = map(int, input().split())
    v.append(vi)
    w.append(wi)
dp = [[0] * (k + 1) for i in range(n + 1)]
for i in range(1, n + 1):
    for j in range(1, k + 1):
        dp[i][j] = dp[i - 1][j]
        if j >= v[i]:
            dp[i][j] = max(dp[i][j],
                           dp[i][j - v[i]] + w[i])
print(dp[n][k])
```

##### 整数划分

###### 递归

```python
from functools import lru_cache
n, mod = int(input()), int(1e9) + 7
@lru_cache()
def count(n, m):
    if n == 0:
        return 1
    if m == 0:
        return 0
    cnt = count(n, m - 1)
    if n >= m:
        cnt = (cnt + count(n - m, m)) % mod
    return cnt
print(count(n, n))
```

###### dp

```python
n, mod = int(input()), int(1e9) + 7
dp = [[1] * (n + 1)] + [[0] * (n + 1) for i in range(n)]
for i in range(1, n + 1):
    for j in range(1, n + 1):
        dp[i][j] = dp[i][j - 1]
        if i >= j:
            dp[i][j] = (dp[i][j] + dp[i - j][j]) % mod
print(dp[n][n])
```

#### 多重

##### 朴素

```python
n, k = map(int, input().split())
v, w, s = [0], [0], [0]
for i in range(n):
    vi, wi, si = map(int, input().split())
    v.append(vi)
    w.append(wi)
    s.append(si)
dp = [[0] * (k + 1) for i in range(n + 1)]
for i in range(1, n + 1):
    for j in range(1, k + 1):
        m = min(s[i], j // v[i])
        for l in range(m + 1):
            dp[i][j] = max(dp[i][j],
                           dp[i - 1][j - v[i] * l] + w[i] * l)
print(dp[n][k])
```

##### 二进制优化

```python
from math import log2
n, k = map(int, input().split())
v, w = [0], [0]
for i in range(n):
    vi, wi, si = map(int, input().split())
    ki = int(log2(si + 1)) - 1
    v.append(vi)
    w.append(wi)
    for j in range(1, ki + 1):
        v.append(v[-1] << 1)
        w.append(w[-1] << 1)
    m = si + 1 - (1 << ki + 1)
    v.append(vi * m)
    w.append(wi * m)
n = len(v) - 1
dp = [[0] * (k + 1) for i in range(n + 1)]
for i in range(1, n + 1):
    for j in range(1, k + 1):
        dp[i][j] = dp[i - 1][j]
        if j >= v[i]:
            dp[i][j] = max(dp[i][j],
                           dp[i - 1][j - v[i]] + w[i])
print(dp[n][k])
```

#### 分组

```python
n, k = map(int, input().split())
vw = [[(0, 0)]]
for i in range(1, n + 1):
    vw.append([(0, 0)])
    m = int(input())
    for j in range(m):
        vw[i].append(tuple(map(int, input().split())))
dp = [[0] * (k + 1) for i in range(n + 1)]
for i in range(1, n + 1):
    for j in range(1, k + 1):
        for v, w in vw[i]:
            if j >= v:
                dp[i][j] = max(dp[i][j],
                               dp[i - 1][j - v] + w)
print(dp[n][k])
```

### 区间

#### 递归

```python
class Cache:
    def __init__(self):
        from sys import setrecursionlimit
        from functools import lru_cache

        setrecursionlimit(int(1e6))

        self.lru_cache = lru_cache

    def func(self):
        return self.lru_cache(maxsize=None)     # return key is function


cache = Cache()
stone_sum = [0]


def query_sum(l, r):
    return stone_sum[r] - stone_sum[l - 1]


@cache.func()
def min_stone(l, r):
    if l == r:
        return 0

    m = float('inf')
    for mid in range(l, r):
        left_cost = min_stone(l, mid) + query_sum(l, mid)
        right_cost = min_stone(mid + 1, r) + query_sum(mid + 1, r)

        m = min(m, left_cost + right_cost)
    return m


if __name__ == '__main__':
    n = int(input())

    for stone in map(int, input().split()):
        stone_sum.append(stone_sum[-1] + stone)

    print(min_stone(1, n))
```

#### dp

```python
stone_sum = [0]


def query_sum(l, r):
    return stone_sum[r] - stone_sum[l - 1]


if __name__ == '__main__':
    n = int(input())

    for stone in map(int, input().split()):
        stone_sum.append(stone_sum[-1] + stone)

    dp = [[0] * (n + 1) for _ in range(n + 1)]

    for length in range(2, n + 1):
        for l in range(1, n - length + 2):
            r = l + length - 1
            m = float('inf')
            
            for mid in range(l, r):
                left = dp[l][mid] + query_sum(l, mid)
                right = dp[mid + 1][r] + query_sum(mid + 1, r)
                
                m = min(m, left + right)
            
            dp[l][r] = m
            
    print(dp[1][n])
```

### 树型

等价于记忆化搜索

```python
class Cache:
    def __init__(self, limit=int(1e5)):
        from sys import setrecursionlimit
        from functools import lru_cache

        setrecursionlimit(limit)

        self.lru_cache = lru_cache

    def func(self):
        return self.lru_cache()     # return key is function


if __name__ == '__main__':
    cache = Cache()

    n = int(input())
    g = [[] for i in range(n + 1)]
    w = [0] + [int(input()) for i in range(n)]
    root = set(range(1, n + 1))
    for i in range(n - 1):
        s, f = map(int, input().split())
        g[f].append(s)
        root.discard(s)
    root = root.pop()

    @cache.func()
    def attend(k):     # k root(attend) max joy
        s = w[k]
        for i in g[k]:
            s += max(0, absent(i))
        return s
    
    dfs = lambda k: max(attend(k), absent(k))

    @cache.func()
    def absent(k):      # k root(absent) max joy
        s = 0
        for i in g[k]:
            s += max(0, dfs(i))
        return s

    print(dfs(root))
```

### 状态压缩

#### 集合

```python
n, m, k = map(int, input().split())
N = 1 << m
bags = [0] * n
for i in range(n):
    it = map(int, input().split())
    for j in it:
        bags[i] |= 1 << j - 1
dp = [101] * N
dp[0] = 0
for st in range(N):
    for bag in bags:
        dp[st | bag] = min(dp[st | bag], dp[st] + 1)
if dp[N - 1] > 100:
    print(-1)
else:
    print(dp[N - 1])
```

#### 棋盘

```python
while True:
    n, m = map(int, input().split())
    if not n and not m:
        break
    ST = 1 << n
    law = [True] * ST
    for st in range(ST):
        cnt = 0
        for k in range(n):
            if st >> k & 1:
                if cnt & 1:
                    law[st] = False
                    break
                cnt = 0
            else:
                cnt += 1
        if cnt & 1:
            law[st] = False
    dp = [[0] * ST for i in range(m + 1)]
    dp[0][0] = 1
    for i in range(1, m + 1):
        for st in range(ST):
            for last in range(ST):
                if not (st & last) and law[st | last]:
                    dp[i][st] += dp[i - 1][last]
    print(dp[m][0])
```

### 数位

```python
N = 10
f = [[1] * N] + [[0] * N for _ in range(N - 1)]
for k in range(1, N):
    for n in range(N):
        for i in range(n, N):
            f[k][n] += f[k - 1][i]

def count(n):   # n >= 0, [0, n]
    nums = [0] + [int(x) for x in str(n)]
    size = len(nums) - 1
    def dfs(k):
        if k < 0:
            return 1
        p = size - k
        if nums[p - 1] > nums[p]:
            return 0
        cnt = 0
        for i in range(nums[p - 1], nums[p]):
            cnt += f[k][i]
        return cnt + dfs(k - 1)
    return dfs(size - 1)

while True:
    try:
        l, r = map(int, input().split())
        print(count(r) - count(l - 1))
    except:
        break
```

## 贪心算法

### 绝对值不等式

$$
\sum_{i=0}^{n-1}{x-a_i}
$$

```python
def abs_min(a):
    n = len(a)
    b = sorted(a)
    l, r = 0, n - 1
    s = 0
    while l < r:
        s += b[r] - b[l]
        l += 1
        r -= 1
    return s
```

#### 糖果传递

```python
n = int(input())
a = [int(input()) for i in range(n)]
av = sum(a) // n
b = [0]
for i in range(1, n):
    b.append(b[i - 1] + av - a[i])
print(abs_max(n, b))
```

### 区间选点

```python
n = int(input())
intervals = sorted([tuple(map(int, input().split())) for i in range(n)], key = lambda t: t[1])
r, cnt = intervals[0][1], 1
for i in range(1, n):
    l = intervals[i][0]
    if l > r:
        r = intervals[i][1]
        cnt += 1
print(cnt)
```
