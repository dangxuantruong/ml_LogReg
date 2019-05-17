import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

LR = LogisticRegression(solver='saga')

traindata = np.load("drive/My Drive/ML_colab/ML_Sarcasm_Headline/traindata.npy")
testdata = np.load("drive/My Drive/ML_colab/ML_Sarcasm_Headline/testdata.npy")


x_train = traindata[0: , :-1]
y_train = traindata[0: , -1]

x_test = testdata[0: , :-1]
y_test = testdata[0: , -1]

LR.fit(x_train, y_train)
_predict = LR.predict(x_test)

f1 = f1_score(y_test, _predict)
accuracy = accuracy_score(y_test, _predict)
precision = precision_score(y_test, _predict)
recall = recall_score(y_test, _predict)

print("f1_score:", f1)
print("precision:", precision)
print("recall:", recall)
print("accuracy:", accuracy)

trainscore = LR.score(x_train, y_train)
testscore = LR.score(x_test, y_test)
print(trainscore)
print(testscore)