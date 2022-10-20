import pickle
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# ROOT

knn1 = pickle.load(open('KNN_digit_letter.sav', 'rb'))
dt1 = pickle.load(open('DT_digit_letter.sav', 'rb'))
svm1 = pickle.load(open('SVM_digit_letter.sav', 'rb'))

# IF DIGIT

knn11 = pickle.load(open('KNN_even_odd.sav', 'rb'))
dt11 = pickle.load(open('DT_even_odd.sav', 'rb'))
svm11 = pickle.load(open('SVM_even_odd.sav', 'rb'))

knn111 = pickle.load(open('KNN_even_values.sav', 'rb'))
dt111 = pickle.load(open('DT_even_values.sav', 'rb'))

knn112 = pickle.load(open('KNN_odd_values.sav', 'rb'))
dt112 = pickle.load(open('DT_odd_values.sav', 'rb'))

# IF LETTER
knn12 = pickle.load(open('KNN_upper_lower.sav', 'rb'))
dt12 = pickle.load(open('DT_upper_lower.sav', 'rb'))
svm12 = pickle.load(open('SVM_upper_lower.sav', 'rb'))

knn121 = pickle.load(open('KNN_upper_values.sav', 'rb'))
dt121 = pickle.load(open('DT_upper_values.sav', 'rb'))

knn122 = pickle.load(open('KNN_lower_values.sav', 'rb'))
dt122 = pickle.load(open('DT_lower_values.sav', 'rb'))


def weighted_decision(knn, dt, svm):
    decision = {45: knn, 35: svm, 20: dt}
    label1 = list()
    label2 = list()
    val1 = list()
    val2 = list()

    for key, value in decision.items():
        if value in label1:
            val1.append(key)
            label1.append(value)
        if value in label2:
            val2.append(key)
            label2.append(value)
        else:
            val1.append(key)
            label1.append(value)
    if sum(val1) >= sum(val2):
        return label1[0]
    else:
        return label2[0]




def ensemble(x_pred: np.ndarray):
    # LEVEL 1
    pred100 = knn1.predict(x_pred)
    pred101 = dt1.predict(x_pred)
    pred110 = svm11.predict(x_pred)
    level1 = weighted_decision(pred100,pred101,pred110)

    # LEVEL 2
    pred200 = knn11.predict(x_pred)
    pred201 = dt11.predict(x_pred)
    pred210 = svm11.predict(x_pred)
    level2 = weighted_decision(pred200, pred201, pred210)

    # LEVEL 3
    pred300 = knn111.predict(x_pred)
    pred301 = dt111.predict(x_pred)
    level3 = weighted_decision(pred300, pred301)

    # LEVEL 4
    pred400 = knn112.predict(x_pred)
    pred401 = dt112.predict(x_pred)
    level3 = weighted_decision(pred400, pred401)

    # LEVEL 5
    pred500 = knn12.predict(x_pred)
    pred501 = dt12.predict(x_pred)
    pred510 = svm12.predict(x_pred)
    level5 = weighted_decision(pred500, pred501, pred510)

    # LEVEL 6
    pred600 = knn121.predict(x_pred)
    pred601 = dt121.predict(x_pred)
    level6 = weighted_decision(pred600, pred601)

    # LEVEL 7
    pred700 = knn122.predict(x_pred)
    pred701 = dt122.predict(x_pred)
    level7 = weighted_decision(pred700, pred701)













