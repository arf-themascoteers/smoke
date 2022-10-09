import numpy as np
import torch

file = open("data/data.csv","w")
col = 300
x = np.random.rand(1000,col)
weights = np.random.rand(col)
bias = np.random.rand(1)[0]

for j in range(col):
    file.write(f"x{j},")
file.write("y\n")
for i in range(x.shape[0]):
    for j in range(col):
        file.write(f"{x[i][j]},")

    y = np.sum(x[i]*weights)/(col/2) + bias
    file.write(f"{y}\n")

file.close()
print("done")