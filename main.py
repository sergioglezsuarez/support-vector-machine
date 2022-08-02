from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from os import listdir

df = pd.read_csv("datos_practica6.csv", sep=";", decimal=".")
X = np.array(df[["x1", "x2"]])
y = np.array(df["y"])

diccionario = {0:"brown", 1:"yellow", 2:"purple"}
colores = [diccionario[i] for i in y]

model1 = svm.SVC(kernel="linear", C=0.01).fit(X, y)
model2 = svm.SVC(kernel="linear", C=1).fit(X, y)
model3 = svm.SVC(kernel="linear", C=10).fit(X, y)

values1, values2, values3, values4, values5, values6, values7, values8, values9 = ([], [], [], [], [], [], [], [], [])

for x in X:
    values1.append(((model1.coef_[0][0]*x[0])/model1.coef_[0][1])+(model1.intercept_[0]/model1.coef_[0][1]))
    values4.append(((model2.coef_[0][0]*x[0])/model2.coef_[0][1])+(model2.intercept_[0]/model2.coef_[0][1]))
    values7.append(((model3.coef_[0][0]*x[0])/model3.coef_[0][1])+(model3.intercept_[0]/model3.coef_[0][1]))
    values2.append(((model1.coef_[1][0]*x[0])/model1.coef_[1][1])+(model1.intercept_[1]/model1.coef_[1][1]))
    values5.append(((model2.coef_[1][0]*x[0])/model2.coef_[1][1])+(model2.intercept_[1]/model2.coef_[1][1]))
    values8.append(((model3.coef_[1][0]*x[0])/model3.coef_[1][1])+(model3.intercept_[1]/model3.coef_[1][1]))
    values3.append(((model1.coef_[2][0]*x[0])/model1.coef_[2][1])+(model1.intercept_[2]/model1.coef_[2][1]))
    values6.append(((model2.coef_[2][0]*x[0])/model2.coef_[2][1])+(model2.intercept_[2]/model2.coef_[2][1]))
    values9.append(((model3.coef_[2][0]*x[0])/model3.coef_[2][1])+(model3.intercept_[2]/model3.coef_[2][1]))

fig, axs = plt.subplots()
axs.plot(pd.DataFrame(X)[0], values1, color="b", label='Superficie de decisión 1 C=0.01')
axs.plot(pd.DataFrame(X)[0], values2, color="b", label='Superficie de decisión 2 C=0.01')
axs.plot(pd.DataFrame(X)[0], values3, color="b", label='Superficie de decisión 3 C=0.01')
axs.scatter(pd.DataFrame(X)[0], pd.DataFrame(X)[1], c=colores)
axs.scatter(pd.DataFrame(model1.support_vectors_)[0], pd.DataFrame(model1.support_vectors_)[1], color="r", marker="+")
axs.set_ylim(bottom=-15, top=15)
axs.legend()
plt.show()

fig, axs = plt.subplots()
axs.plot(pd.DataFrame(X)[0], values4, color="r", label='Superficie de decisión 1 C=1')
axs.plot(pd.DataFrame(X)[0], values5, color="r", label='Superficie de decisión 2 C=1')
axs.plot(pd.DataFrame(X)[0], values6, color="r", label='Superficie de decisión 3 C=1')
axs.scatter(pd.DataFrame(X)[0], pd.DataFrame(X)[1], c=colores)
axs.scatter(pd.DataFrame(model2.support_vectors_)[0], pd.DataFrame(model2.support_vectors_)[1], color="r", marker="+")
axs.set_ylim(bottom=-15, top=15)
axs.legend()
plt.show()

fig, axs = plt.subplots()
axs.plot(pd.DataFrame(X)[0], values7, color="g", label='Superficie de decisión 1 C=10')
axs.plot(pd.DataFrame(X)[0], values8, color="g", label='Superficie de decisión 2 C=10')
axs.plot(pd.DataFrame(X)[0], values9, color="g", label='Superficie de decisión 3 C=10')
axs.scatter(pd.DataFrame(X)[0], pd.DataFrame(X)[1], c=colores)
axs.scatter(pd.DataFrame(model3.support_vectors_)[0], pd.DataFrame(model3.support_vectors_)[1], color="r", marker="+")
axs.set_ylim(bottom=-15, top=15)
axs.legend()
plt.show()

# A medida que aumenta C, hay menos vectores soporte y la clasificación es un poco mejor, notándose en las muestras marrones y amarillas principalmente.

train = pd.read_csv("CelebA-10K-train.csv")
train, valid = train_test_split(train, train_size=0.8)
Xtrain = train.iloc[:, 2:]
ytrain = train["Gender"]
Xvalid = valid.iloc[:, 2:]
yvalid = valid["Gender"]
test = pd.read_csv("CelebA-10K-test.csv")
Xtest = test.iloc[:, 2:]
ytest = test["Gender"]

results = []
maximo = 0
combinations = product(["linear", "rbf", "poly"], [0.001, 0.1, 1, 10, 25])
best = []

for combination in combinations:
    model = svm.SVC(kernel=combination[0], C=combination[1]).fit(Xtrain, ytrain)
    acc = accuracy_score(yvalid, model.predict(Xvalid))
    if acc > maximo:
        maximo = acc
        best = [combination[0], combination[1]]
    results.append([combination[0], combination[1], acc])
    print(combination[0], combination[1], acc)

print("Mejor combinación:", best[0], best[1])

model = svm.SVC(kernel=best[0], C=best[1]).fit(Xtrain, ytrain)
test_acc = accuracy_score(ytest, model.predict(Xtest))
print("Test accuracy:", test_acc)

lista = listdir("ImagenesParaClasificar")

images = test[test["Image_name"].isin(lista)]

preds = model.predict(images.iloc[:, 2:])
print(pd.DataFrame({"ytrue": images["Gender"], "pred": preds}))
print("Clasificación accuracy:", accuracy_score(preds, images["Gender"]))
print("Muestras mal clasificadas:")
for i in range(len(images)):
    if preds[i]!=images["Gender"].iloc[i]:
        print(images.iloc[i,0])