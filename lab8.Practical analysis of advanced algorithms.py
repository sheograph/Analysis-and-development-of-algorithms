import matplotlib.pyplot as plt
import pandas as pd
import timeit
from pympler import asizeof
import warnings
import random

warnings.filterwarnings(action='ignore')
random.seed(0)

#Implementing the Finite Automata algorithm for pattern matching.
txt = "AABAACAADAABAAABAA"
pat = "AABA"

NO_OF_CHARS = 256

def getNextState(pattern, M, state, x):
    '''
    calculate the next state
    '''

    # If the character c is same as next character
    # in pattern, then simply increment state

    if state < M and x == ord(pattern[state]):
        return state + 1

    i = 0
    # ns stores the result which is next state

    # ns finally contains the longest prefix
    # which is also suffix in "pat[0..state-1]c"

    # Start from the largest possible value and
    # stop when you find a prefix which is also suffix
    for ns in range(state, 0, -1):
        if ord(pattern[ns - 1]) == x:
            while (i < ns - 1):
                if pattern[i] != pattern[state - ns + 1 + i]:
                    break
                i += 1
            if i == ns - 1:
                return ns
    return 0


def computeTF(pattern, M):
    '''
    This function builds the TF table which
    represents Finite Automata for a given pattern
    '''
    global NO_OF_CHARS

    TF = [[0 for i in range(NO_OF_CHARS)] \
          for _ in range(M + 1)]

    for state in range(M + 1):
        for x in range(NO_OF_CHARS):
            z = getNextState(pattern, M, state, x)
            TF[state][x] = z

    return TF


def search_Finite_Automaton(pattern, text):
    '''
    Prints all occurrences of pat in txt
    '''
    global NO_OF_CHARS
    M = len(pattern)
    N = len(text)
    TF = computeTF(pattern, M)
    iterations = 0
    index_list = []

    # Process txt over FA.
    state = 0
    for i in range(N):
        iterations += 1
        state = TF[state][ord(text[i])]
        if state == M:
            index_list.append(i)
            print("Pattern found at index: {}". \
                  format(i - M + 1))
    return index_list, iterations

txt = "AABAACAADAABAAABAA"
pat = "AABA"

index_list, iterations = search_Finite_Automaton(
    pattern=pat,
    text=txt
)

print('Number of Iterations:', iterations)
print('Index List:', index_list)


#Implementing the Knuth-Morris-Pratt algorithm for pattern matching.
def return_prefix(pat1):
    pattern_len = len(pat1)
    prefix = [0] * pattern_len
    i = 0
    j = 1

    while j < pattern_len:
        if pat1[i] == pat1[j]:
            prefix[j] = i + 1
            i += 1
            j += 1

        elif i:
            i = prefix[i - 1]

        else:
            prefix[j] = 0
            j += 1
    return prefix


def kmp_search(text, pattern):
    pattern_len = len(pattern)
    text_len = len(text)
    prefix = return_prefix(pattern)
    iterations = 0
    index_list = []
    i = 0
    j = 0

    while i < text_len and j < pattern_len:
        iterations += 1
        if text[i] == pattern[j]:
            if j == pattern_len - 1:
                index_list.append(i - pattern_len + 1)
                j = 0

            else:
                j += 1

            i += 1
        elif j:
            j = prefix[j - 1]

        else:
            i += 1
    return index_list, iterations


pat1 = 'abc'
txt1 = 'abcabeabdabc'

index_list, iterations = kmp_search(
    pattern=pat1,
    text=txt1
)

print('Number of Iterations:', iterations)
print('Index List:', index_list)

df = pd.DataFrame()
pattern = 'abc'
letters = 'abcde'
index = 0

for i in range(0, 101, 1):
    text = ''.join(random.choice(letters) for j in range(i))
    df.loc[index, 'Finite_Automaton_time'] = timeit.timeit(
        'search_Finite_Automaton(text, pattern)',
        setup='from __main__ import search_Finite_Automaton, pattern, text',
        number=3)

    df.loc[index, 'Finite_Automaton_memory'] = asizeof.asizeof(search_Finite_Automaton(text, pattern))
    _, df.loc[index, 'Finite_Automaton_iterations'] = search_Finite_Automaton(text, pattern)

    df.loc[index, 'kmp_time'] = timeit.timeit(
        'kmp_search(text, pattern)',
        setup='from __main__ import kmp_search, return_prefix, pattern, text',
        number=3)

    df.loc[index, 'kmp_memory'] = asizeof.asizeof(kmp_search(text, pattern))
    _, df.loc[index, 'kmp_iterations'] = kmp_search(text, pattern)
    index = index + 1
    print(i)
df.to_csv('time.csv', index=False)
df.head(10)


def plot(data_1, data_2, label, ylabel):
    plt.figure(figsize=(10, 5))
    plt.plot(data_1, label='Finite Automata algorithm')
    plt.plot(data_2, label='Knuth-Morris-Pratt Algorithm')

    ax = plt.gca()
    ax.set_title(label)
    ax.legend()
    plt.xlabel('Dimension')
    plt.ylabel(ylabel)
    plt.show()


print(df)
plot(df['Finite_Automaton_time'], df['kmp_time'], 'Comparison of Algorithm Execution Time', 'Time')