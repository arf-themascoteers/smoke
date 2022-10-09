import my_dataset
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score

def train():
    ds = my_dataset.MyDataset(is_train=True)
    x = ds.get_x()
    y = ds.get_y()
    reg = LinearRegression().fit(x,y)

    print("Train done")

    pickle.dump(reg, open("models/linear","wb"))

def test():
    reg = pickle.load(open('models/linear', 'rb'))
    ds = my_dataset.MyDataset(is_train=False)
    x = ds.get_x()
    y = ds.get_y()

    y_hat = reg.predict(x)

    print(r2_score(y, y_hat))

    for i in range(10):
        a_y = y[i]
        a_y_hat = y_hat[i]
        print(f"{a_y:.3}\t\t{a_y_hat:.3}")

train()
test()