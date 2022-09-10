import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
import scipy as sp


# f(v)=const
def Const(N, runs):
    ts1 = np.zeros(N)
    for i in range(0, runs):
        ts1_ = np.array([])
        time_stamp_1 = np.array([datetime.datetime.now().timestamp()])
        for k in range(1, N + 1):
            arr1 = np.random.sample(k)
            arr1.fill(1)
            # for j in range(0, k):
            #     arr1[j] = 1
            time_stamp_1 = np.append(time_stamp_1, datetime.datetime.now().timestamp())
            ts1_ = np.append(ts1_, time_stamp_1[k] - time_stamp_1[k - 1])
        ts1 = ts1 + ts1_
    ts1 = ts1 / runs
    return ts1


# f(v)=SUM
def Sum(N, runs):
    ts2 = np.zeros(N)
    for i in range(0, runs):
        ts2_ = np.array([])
        sum_arr2 = 0
        time_stamp_2 = np.array([datetime.datetime.now().timestamp()])
        for k in range(1, N + 1):
            arr2 = np.random.sample(k)
            sum_arr2 = np.sum(arr2)
            # for j in range(0, k):
            #     sum_arr2 = sum_arr2 + arr2[j]
            time_stamp_2 = np.append(time_stamp_2, datetime.datetime.now().timestamp())
            ts2_ = np.append(ts2_, time_stamp_2[k] - time_stamp_2[k - 1])
        ts2 = ts2 + ts2_
    ts2 = ts2 / runs
    return ts2


# f(v)=PROD
def Prod(N, runs):
    ts3 = np.zeros(N)
    for i in range(0, runs):
        ts3_ = np.array([])
        prod_arr3 = 1
        time_stamp_3 = np.array([datetime.datetime.now().timestamp()])
        for k in range(1, N + 1):
            arr3 = np.random.sample(k)
            prod_arr3 = np.prod(arr3)
            # for j in range(0, k):
            #     prod_arr3 = prod_arr3 * arr3[j]
            time_stamp_3 = np.append(time_stamp_3, datetime.datetime.now().timestamp())
            ts3_ = np.append(ts3_, time_stamp_3[k] - time_stamp_3[k - 1])
        ts3 = ts3 + ts3_
    ts3 = ts3 / runs
    return ts3


# P(x)=SUM(v_k*x^(k-1)), x=1.5
def Polynom_Naive(N, runs, x):
    ts4 = np.zeros(N)
    for i in range(0, runs):
        ts4_ = np.array([])
        time_stamp_4 = np.array([datetime.datetime.now().timestamp()])
        for k in range(1, N + 1):
            x_power = x
            arr4 = np.random.sample(k)
            sum_arr4 = arr4[0]
            for j in range(1, k):
                # sum_arr4 = sum_arr4 + arr4[j] * np.power(x, j)
                sum_arr4 = sum_arr4 + (arr4[j] * x_power)
                x_power = x_power * x
            time_stamp_4 = np.append(time_stamp_4, datetime.datetime.now().timestamp())
            ts4_ = np.append(ts4_, time_stamp_4[k] - time_stamp_4[k - 1])
        ts4 = ts4 + ts4_
    ts4 = ts4 / runs
    return ts4


# P(x)=v1+x*(v2+x*(v3+...))
def Polynom_Horner(N, runs, x):
    ts4 = np.zeros(N)
    for i in range(0, runs):
        ts4_ = np.array([])
        time_stamp_4 = np.array([datetime.datetime.now().timestamp()])
        arr4 = np.random.sample(1)
        sum_arr4 = arr4[0]
        time_stamp_4 = np.append(time_stamp_4, datetime.datetime.now().timestamp())
        ts4_ = np.append(ts4_, time_stamp_4[1] - time_stamp_4[0])
        for k in range(2, N + 1):
            arr4 = np.random.sample(k)
            sum_arr4 = arr4[k - 1]
            for j in range(k - 1, 0, -1):
                sum_arr4 = sum_arr4 * x
                sum_arr4 = sum_arr4 + arr4[j - 1]
            time_stamp_4 = np.append(time_stamp_4, datetime.datetime.now().timestamp())
            ts4_ = np.append(ts4_, time_stamp_4[k] - time_stamp_4[k - 1])
        ts4 = ts4 + ts4_
    ts4 = ts4 / runs
    return ts4


# bubble_sort
ts5 = np.array([])
prod_arr5 = 1
time_stamp_5 = np.array([datetime.datetime.now().timestamp()])


def bubble_sort(arr):
    def swap(i, j):
        arr[i], arr[j] = arr[j], arr[i]

    n = len(arr)
    swapped = True

    x = -1
    while swapped:
        swapped = False
        x = x + 1
        for i in range(1, n - x):
            if arr[i - 1] > arr[i]:
                swap(i - 1, i)
                swapped = True


# quick_sort
def partition(array, begin, end):
    pivot_idx = begin
    for i in xrange(begin + 1, end + 1):
        if array[i] <= array[begin]:
            pivot_idx += 1
            array[i], array[pivot_idx] = array[pivot_idx], array[i]
    array[pivot_idx], array[begin] = array[begin], array[pivot_idx]
    return pivot_idx


def quick_sort_recursion(array, begin, end):
    if begin >= end:
        return
    pivot_idx = partition(array, begin, end)
    quick_sort_recursion(array, begin, pivot_idx - 1)
    quick_sort_recursion(array, pivot_idx + 1, end)


def quick_sort(array, begin=0, end=None):
    if end is None:
        end = len(array) - 1

    return quick_sort_recursion(array, begin, end)


# timsort
MIN_MERGE = 32


def calcMinRun(n):
    """Returns the minimum length of a
    run from 23 - 64 so that
    the len(array)/minrun is less than or
    equal to a power of 2.

    e.g. 1=>1, ..., 63=>63, 64=>32, 65=>33,
    ..., 127=>64, 128=>32, ...
    """
    r = 0
    while n >= MIN_MERGE:
        r |= n & 1
        n >>= 1
    return n + r


# This function sorts array from left index to
# to right index which is of size atmost RUN
def insertionSort(arr, left, right):
    for i in range(left + 1, right + 1):
        j = i
        while j > left and arr[j] < arr[j - 1]:
            arr[j], arr[j - 1] = arr[j - 1], arr[j]
            j -= 1


# Merge function merges the sorted runs
def merge(arr, l, m, r):
    # original array is broken in two parts
    # left and right array
    len1, len2 = m - l + 1, r - m
    left, right = [], []
    for i in range(0, len1):
        left.append(arr[l + i])
    for i in range(0, len2):
        right.append(arr[m + 1 + i])

    i, j, k = 0, 0, l

    # after comparing, we merge those two array
    # in larger sub array
    while i < len1 and j < len2:
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1

        else:
            arr[k] = right[j]
            j += 1

        k += 1

    # Copy remaining elements of left, if any
    while i < len1:
        arr[k] = left[i]
        k += 1
        i += 1

    # Copy remaining element of right, if any
    while j < len2:
        arr[k] = right[j]
        k += 1
        j += 1


# Iterative Timsort function to sort the
# array[0...n-1] (similar to merge sort)
def timSort(arr):
    n = len(arr)
    minRun = calcMinRun(n)

    # Sort individual subarrays of size RUN
    for start in range(0, n, minRun):
        end = min(start + minRun - 1, n - 1)
        insertionSort(arr, start, end)

    # Start merging from size RUN (or 32). It will merge
    # to form size 64, then 128, 256 and so on ....
    size = minRun
    while size < n:

        # Pick starting point of left sub array. We
        # are going to merge arr[left..left+size-1]
        # and arr[left+size, left+2*size-1]
        # After every merge, we increase left by 2*size
        for left in range(0, n, 2 * size):

            # Find ending point of left sub array
            # mid+1 is starting point of right sub array
            mid = min(n - 1, left + size - 1)
            right = min((left + 2 * size - 1), (n - 1))

            # Merge sub array arr[left.....mid] &
            # arr[mid+1....right]
            if mid < right:
                merge(arr, left, mid, right)

        size = 2 * size


def Matrix_Prod(N, runs):
    start_time = time.time()
    ts = np.zeros(N)
    for i in range(0, runs):
        ts_ = np.array([])
        time_stamp = np.array([datetime.datetime.now().timestamp()])
        for k in range(1, N + 1):
            A = np.random.random((k, k))
            B = np.random.random((k, k))
            C = np.zeros((k, k))
            for i in range(len(A)):
                for j in range(len(B)):
                    for l in range(k):
                        C[i, j] = C[i, j] + A[i, l] * B[l, j]
            # matrix_prod = A.dot(B)
            time_stamp = np.append(time_stamp, datetime.datetime.now().timestamp())
            ts_ = np.append(ts_, time_stamp[k] - time_stamp[k - 1])
            print(k, 'iteration\t', time.time() - start_time, 'seconds')
        ts = ts + ts_
    ts = ts / runs
    return ts


def plot_LSM(N, n, ts, str, d):  # Least Squares Method for linear func
    mx = n.sum() / N
    my = ts.sum() / N
    a2 = np.dot(n.T, n) / N
    a11 = np.dot(n.T, ts) / N

    kk = (a11 - mx * my) / (a2 - mx ** 2)
    bb = my - kk * mx
    ff = np.array([kk * z + bb for z in range(N)])

    fig, ax = plt.subplots()
    ax.set_xlabel('Vector dimension v')
    ax.set_ylabel('Time(seconds)')
    fig.suptitle(str)
    plt.plot(n, ts, label='Experimental value')
    plt.plot(ff, label='Theoretical value')
    ax.legend()
    plt.grid(True)
    plt.show()


def plot_advanced_LSM(N, n, ts, str, d):  # Least Squares Method for any integer degree
    fx = np.linspace(n[0], n[-1] + 10, 1000)
    fp, residuals, rank, sv, rcond = np.polyfit(n, ts, d, full=True)
    f = sp.poly1d(fp)  # polynomial function
    fig, ax = plt.subplots()
    ax.set_xlabel('Vector dimension v')
    ax.set_ylabel('Time(seconds)')
    plt.suptitle(str)
    plt.plot(n, ts, label='Experimental value')
    plt.plot(fx, f(fx), label="Theoretical value O(n^%d)" % f.order)
    plt.legend()
    plt.grid()
    plt.savefig(f'{str}.png', dpi=300)
    plt.show()


N = 2000  # Vector dimension
n = np.linspace(1, N, N)
x = 1.5
runs = 1  # Number of launches

d = 0
ts1 = Const(N, runs)
ts2 = Sum(N, runs)
ts3 = Prod(N, runs)
plot_advanced_LSM(N, n, ts1, 'f(v)=CONST', d)
plot_advanced_LSM(N, n, ts2, 'f(v)=SUM', d)
plot_advanced_LSM(N, n, ts3, 'f(v)=PROD', d)

d = 1
ts4_1 = Polynom_Naive(N, runs, x)
ts4_2 = Polynom_Horner(N, runs, x)
plot_advanced_LSM(N, n, ts4_1, 'Polynom Naive', d)
plot_advanced_LSM(N, n, ts4_2, 'Polynom Horner', d)


N = 50
n = np.linspace(1, N, N)
d = 3
ts8 = Matrix_Prod(N, runs)
plot_advanced_LSM(N, n, ts8, 'Matrix Product', d)
