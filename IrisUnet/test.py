a, b, c, d = map(int, input().split())

eps = 0.001
ans = 0

def fun(x):
    return a * (x ** 3) + b * (x ** 2) + c * x + d


def pd(m, n):
    if fun(m) * fun(n) < 0:
        return 1
    return 0



for i in range(-99, 100):
    # print(f'f({i})={li[j]}')
    if fun(i) == 0:
        print(f'{i:.02f}', end=' ')
        ans += 1
    elif fun(i) * fun(i - 1) < 0:
        low = i - 1
        high = i
        while low + eps < high or fun(high) != 0:
            mid = (low + high) / 2
            if pd(mid, high):
                low = mid
            else:
                high = mid
        print(f'{high:.02f}', end=' ')
        ans += 1
    if ans == 3:
        break



