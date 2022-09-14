import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit


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


def Bubble_Sort(N, runs):
    def bubblesort(l):
        last_element = len(l) - 1
        for x in range(0, last_element):
            for y in range(0, last_element):
                if l[y] > l[y + 1]:
                    l[y], l[y + 1] = l[y + 1], l[y]
        return l
    ts5 = np.zeros(N)
    for i in range(0, runs):
        ts5_ = np.array([])
        time_stamp_5 = np.array([datetime.datetime.now().timestamp()])
        for k in range(1, N + 1):
            arr5 = np.random.sample(k)
            arr5 = bubblesort(arr5)
            time_stamp_5 = np.append(time_stamp_5, datetime.datetime.now().timestamp())
            ts5_ = np.append(ts5_, time_stamp_5[k] - time_stamp_5[k - 1])
        ts5 = ts5 + ts5_
    ts5 = ts5 / runs
    return ts5


def Quick_Sort(N, runs):
    def quicksort(l1):
        if len(l1) > 1:
            x = l1[random.randint(0, len(l1) - 1)]
            low = [u for u in l1 if u < x]
            eq = [u for u in l1 if u == x]
            high = [u for u in l1 if u > x]
            l1 = quicksort(low) + eq + quicksort(high)
        return l1
    ts6 = np.zeros(N)
    for i in range(0,runs):
        ts6_ = np.array([])
        time_stamp_6 = np.array([datetime.datetime.now().timestamp()])
        for k in range(1, N + 1):
            arr6 = np.random.sample(k)
            arr6 = quicksort(arr6)
            time_stamp_6 = np.append(time_stamp_6, datetime.datetime.now().timestamp())
            ts6_ = np.append(ts6_, time_stamp_6[k] - time_stamp_6[k - 1])
        ts6 = ts6 + ts6_
    ts6 = ts6 / runs
    return ts6


def Tim_Sort(N, runs):
    minrun = 32

    def InsSort(arr, start, end):
        for i in range(start + 1, end + 1):
            elem = arr[i]
            j = i - 1
            while j >= start and elem < arr[j]:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = elem
        return arr

    def Merge(arr, start, mid, end):
        if mid == end:
            return arr
        first = arr[start:mid + 1]
        last = arr[mid + 1:end + 1]
        len1 = mid - start + 1
        len2 = end - mid
        ind1 = 0
        ind2 = 0
        ind = start
        while ind1 < len1 and ind2 < len2:
            if first[ind1] < last[ind2]:
                arr[ind] = first[ind1]
                ind1 += 1
            else:
                arr[ind] = last[ind2]
                ind2 += 1
            ind += 1
        while ind1 < len1:
            arr[ind] = first[ind1]
            ind1 += 1
            ind += 1
        while ind2 < len2:
            arr[ind] = last[ind2]
            ind2 += 1
            ind += 1
        return arr

    def Tim_run(arr):
        n = len(arr)
        for start in range(0, n, minrun):
            end = min(start + minrun - 1, n - 1)
            arr = InsSort(arr, start, end)
        curr_size = minrun
        while curr_size < n:
            for start in range(0, n, curr_size * 2):
                mid = min(n - 1, start + curr_size - 1)
                end = min(n - 1, mid + curr_size)
                arr = Merge(arr, start, mid, end)
            curr_size *= 2
        return arr

    ts7 = np.zeros(N)
    for i in range(0, runs):
        ts7_ = np.array([])
        time_stamp_7 = np.array([datetime.datetime.now().timestamp()])
        for k in range(1, N + 1):
            arr7 = Tim_run(np.random.sample(k))
            time_stamp_7 = np.append(time_stamp_7, datetime.datetime.now().timestamp())
            ts7_ = np.append(ts7_, time_stamp_7[k] - time_stamp_7[k - 1])
        ts7 += ts7_
    ts7 = ts7 / runs
    return ts7


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


def O_1(x, c):
    return [c] * len(x)


def O_n(x, a, c):
    return np.multiply(a, x) + c


def O_nlogn(x, a, c):
    logx = np.log2(x)
    return np.multiply(a, np.array(x) * logx) + c


def O_n2(x, a, c):
    x2 = np.array([x[i] ** 2 for i in range(len(x))])
    return a * x2 + c


def O_n3(x, a, c):
    x3 = np.array([x[i] ** 3 for i in range(len(x))])
    return a * x3 + c


def plot_LSM(N, n, ts, str, complexity):  # Least Squares Method for any integer degree
    popt, pcov = curve_fit(complexity, n, ts)
    fig, ax = plt.subplots()
    ax.set_xlabel('Vector dimension v')
    ax.set_ylabel('Time(seconds)')
    plt.suptitle(str)
    plt.plot(n, ts, label='Experimental value')
    c = "Theoretical value " + complexity.__name__
    plt.plot(n, complexity(n, *popt), label=c)
    plt.legend()
    plt.grid()
    plt.savefig(f'{str}.png')
    plt.show()


N = 2000  # Vector dimension
n = np.linspace(1, N, N)
x = 1.5
runs = 5  # Number of launches

ts1 = Const(N, runs)
plot_LSM(N, n, ts1, 'f(v)=CONST', O_1)
ts2 = Sum(N, runs)
plot_LSM(N, n, ts2, 'f(v)=SUM', O_n)
ts3 = Prod(N, runs)
plot_LSM(N, n, ts3, 'f(v)=PROD', O_n)

ts4_1 = Polynom_Naive(N, runs, x)
plot_LSM(N, n, ts4_1, 'Polynom Naive', O_n)
ts4_2 = Polynom_Horner(N, runs, x)
plot_LSM(N, n, ts4_2, 'Polynom Horner', O_n)

N = 200
n = np.linspace(1, N, N)
ts5 = Bubble_Sort(N, runs)
plot_LSM(N, n, ts5, 'Bubble Sort', O_n2)
ts6 = Quick_Sort(N, runs)
plot_LSM(N, n, ts6, 'Quick Sort', O_nlogn)
ts7 = Tim_Sort(N, runs)
plot_LSM(N, n, ts7, 'Tim Sort', O_nlogn)

N = 100
n = np.linspace(1, N, N)
ts8 = Matrix_Prod(N, runs)
plot_LSM(N, n, ts8, 'Matrix Product', O_n3)

