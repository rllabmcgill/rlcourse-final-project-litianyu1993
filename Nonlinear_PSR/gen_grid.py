import numpy as np
import create_files as cf
from collections import Counter
world = [[0, 1, 1, 2],
         [6, 8, 8, 7],
         [6, 8, 8, 7],
         [4, 5, 5, 3]
         ]
world = np.asarray(world)
samples = []
aos = []
for k in range(0, 100):
    sample = []
    ao = []
    i = 3
    j = 0
    while i != 0 or j != 3:
        temp = np.random.rand(1)
        if temp < 0.25:
            action = 0
            if j != 0:
                j -= 1
        elif temp< 0.5:
            action = 1
            if i != 0:
                i -= 1
        elif temp < 0.75:
            action = 2
            if j != 3:
                j+=1
        else:
            action = 3
            if i != 3:
                i+=1
        ao.append(action*4 + world[i][j])
        sample.append(action)
        sample.append(world[i][j])
        if i ==0 and j ==3:
            sample.append(1)
        else:
            sample.append(0)
    aos.append(np.asarray(ao))
    samples.append(np.asarray(sample))

for i in range(0, len(aos)):
    for j in range(0, len(aos[i])):
        if aos[i][j] > 3 and aos[i][j] < 7:
            aos[i][j] -= 2
            continue
        elif aos[i][j] > 7 and aos[i][j] < 14:
            aos[i][j] -= 3
            continue
        elif aos[i][j] > 14:
            aos[i][j] -= 4
            continue
'''
temp = np.zeros(36)
for i in range(0, len(aos)):
    for j in range(0, len(aos[i])):
        temp[aos[i][j]] += 1
for i in range(0, len(temp)):
    print i
    print temp[i]
'''
