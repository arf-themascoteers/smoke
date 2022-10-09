import numpy as np

file = open("data/data.csv","w")
col = 3
x = np.random.rand(1000,col)


file.write("x1,x2,x3,y\n")
for i in range(x.shape[0]):
    for j in range(col):
        file.write(f"{x[i][j]},")

    y = x[i][0] * 0.2 + x[i][1] * 0.3 + x[i][2] *0.4
    file.write(f"{y}\n")

file.close()