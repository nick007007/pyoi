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