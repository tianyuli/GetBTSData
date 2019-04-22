import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, make_scorer, confusion_matrix, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from itertools import islice
import category_encoders as ce
#from sklearn.preprocessing import OrdinalEncoder

DATA = "./output_lastMonth.csv"


def plotROCCurveBase():
    plt.figure()
    #plt.plot(fpr, tpr, lw=lw)
    plt.plot([0, 100], [0, 100], color='navy', lw=2, linestyle='--')
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    #plt.title(title)

def crossValidationTestAndPlot(estimator, name, data, labels, cvNum = 5, addAverage=True):
    fprs = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    th = []
    
    cv = StratifiedKFold(n_splits=5)
    
    for train, test in cv.split(data, labels):
        fit = estimator.fit(data.iloc[train], labels[train])
        y_predict = fit.predict_proba(data.iloc[test])
        fpr, tpr, thresholds = roc_curve(labels[test], y_predict[:, 1])
        th.append(np.interp(mean_fpr, fpr, thresholds)) 
        #plt.plot(fpr, tpr, lw=2)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        print ("Running...")
        
    tprs[-1][0] = 0.0
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    eer_th, eer = findEERThreshold(th, mean_fpr, mean_tpr)

    if addAverage:
        avgAuc = auc(mean_fpr, mean_tpr)
        estLabel = name + ", AUC: " + str(round(avgAuc, 2))
        addAverageToPlotUsingTPRs(tprs, mean_fpr, estLabel)
        #plt.show()
    return mean_tpr
    
def addAverageToPlotUsingTPRs(tprs, mean_fpr, label):
    tprs[-1][0] = 0.0
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 100.0
    plt.plot(mean_fpr*100, mean_tpr*100, label=label)
    
def findEERThreshold(thresholds, mean_fpr, mean_tpr):
    mean_th = np.mean(thresholds,axis=0)
    eer = brentq(lambda x : 1. - x - interp1d(mean_fpr, mean_tpr)(x), 0., 1.)
    eer_th = interp1d(mean_fpr, mean_th)(eer)
    return eer_th, eer

def encode(data):
    hotEncoder = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True)
    data = hotEncoder.fit_transform(data)
    return data
    
    
def main():
    data_full = pd.read_csv(DATA)
    data= data_full.sample(n=10000)
    

    labels = []
    for index, row in data.iterrows():
    #for index, row in islice(data.iterrows(), LIMIT):
         print (row)
         if float(row["DEP_DELAY"]) > 0:
             labels.append(1)
         else:
             labels.append(0)
    data = data.drop('DEP_DELAY', 1)
    sum = 0
    for l in labels:
         sum += l
    print(labels)
    print (sum/len(labels))
    labels = np.asarray(labels)

    categorical = data.select_dtypes(exclude='number')
    numerical = data.select_dtypes(include='number')
    encoded = encode(categorical)

    data = pd.concat([numerical, encoded], axis=1)
    print(data.head())

    
    
    plotROCCurveBase()
    crossValidationTestAndPlot(LogisticRegression(), "Logistic Regression",
                               data, labels, cvNum = 5, addAverage=True)
    print("LR Done")
    crossValidationTestAndPlot(RandomForestClassifier(), "Random Forest",
                               data, labels,
                               cvNum = 5, addAverage=True)
    print("RF Done")
    crossValidationTestAndPlot(KNeighborsClassifier(), "K-Nearest Neighbors",
                               data, labels,
                               cvNum = 5, addAverage=True)
    print("KNN Done")
    crossValidationTestAndPlot(svm.SVC(kernel='linear', probability=True),
                               "SVM - Linear", data, labels,
                               cvNum = 5, addAverage=True)
    print("SVC-linear Done")
    #crossValidationTestAndPlot(svm.SVC(kernel='poly', probability=True), "SVM - Polynomial",
    #                           data, labels,
    #                           cvNum = 5, addAverage=True)
    #print("SVC Poly Done")
    crossValidationTestAndPlot(svm.SVC(kernel='rbf', probability=True), "SVM - Radial",
                               data, labels,
                               cvNum = 5, addAverage=True)
    print("SVC RBF Done")
    crossValidationTestAndPlot(svm.SVC(kernel='sigmoid', probability=True), "SVM - Sigmoid",
                               data, labels,
                               cvNum = 5, addAverage=True)
    print("SVC Sig Done")
    
    plt.legend()
    plt.show()
    
    
main()
