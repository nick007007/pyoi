def max_fix(s):
    n = len(s)
    max_same = [0] * n              # max same fix of s[: i + 1]

    for i in range(1, n):
        j = i - 1

        while j >= 0:
            m = max_same[j]

            if s[i] == s[m]:
                max_same[i] = m + 1
                break

            j = m - 1

    return max_same


def min_fix(s):
    min_same = max_fix(s)

    for i in range(1, len(s)):
        j = i                       # initial j as i > 0

        while m := min_same[j]:     # ensure j and min same fix > 0
            min_same[i] = m
            j = m - 1               # worst j is 0

    return min_same


def kmp(s, p):
    string = p + '#' + s
    s, p = len(s), len(p)

    max_same = max_fix(string)
    # left = p + 1 + i, right = left + p - 1 = p + 1 + i + p - 1 = 2 * p + i
    return [i for i in range(s - p + 1) if max_same[2 * p + i] == p]
