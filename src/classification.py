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
from sklearn.preprocessing import PolynomialFeatures
#from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import RFECV, SelectKBest, f_classif

DATA = "./output_sampled_1000.csv"


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
        print("Fitting estimator")
        fit = estimator.fit(data.iloc[train], labels[train])
        print("Done fitting")
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

def crossValidation(data, labels):
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
    #    #crossValidationTestAndPlot(svm.SVC(kernel='poly', probability=True), "SVM - Polynomial",
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
    #crossValidationTestAndPlot(svm.SVC(kernel='linear', probability=True),
    #                           "SVM - Linear", data, labels,
    #                           cvNum = 5, addAverage=True)
    #print("SVC-linear Done")

    
    plt.legend()
    plt.show()

def featureSelection(data, labels):
    plotROCCurveBase()

    crossValidationTestAndPlot(LogisticRegression(), "Full Feature Set",
                               data, labels, cvNum = 5, addAverage=True)

    # Create and fit selector
    selector = SelectKBest(k=100)
    selector.fit(data, labels)
    # Get columns to keep
    cols = selector.get_support()
    print(len(cols))
    # Create new dataframe with only desired columns, or overwrite existing
    data = data[data.columns[cols]]
    print(data.shape)

    crossValidationTestAndPlot(LogisticRegression(), "100-Best Features",
                               data, labels, cvNum = 5, addAverage=True)

    print(data.columns.values.tolist())
    

    poly = PolynomialFeatures(interaction_only=True)
    polyData = pd.DataFrame(poly.fit_transform(data), columns=poly.get_feature_names(data.columns))

    crossValidationTestAndPlot(LogisticRegression(), "Features with Interaction",
                               polyData, labels, cvNum = 5, addAverage=True)

    #print ("Interaction Features", poly.get_feature_names(data.columns))

    # Create and fit selector
    selector = SelectKBest(k=100)
    selector.fit(polyData, labels)
    # Get columns to keep
    cols = selector.get_support()
    # Create new dataframe with only desired columns, or overwrite existing
    polyData = polyData[polyData.columns[cols]]

    crossValidationTestAndPlot(LogisticRegression(), "100-Best Features With Interaction",
                               polyData, labels, cvNum = 5, addAverage=True)
    print(polyData.columns.values.tolist())
    #print(polyData.get_feature_names(data.columns))

    poly3 = PolynomialFeatures(degree=2)
    poly3Data= pd.DataFrame(poly3.fit_transform(data), columns=poly3.get_feature_names(data.columns))
    crossValidationTestAndPlot(LogisticRegression(), "Features with Polynomials up to n^2",
                               poly3Data, labels, cvNum = 5, addAverage=True)

    
    # Create and fit selector
    selector = SelectKBest(k=100)
    selector.fit(poly3Data, labels)
    # Get columns to keep
    cols = selector.get_support()
    # Create new dataframe with only desired columns, or overwrite existing
    poly3Data = poly3Data[poly3Data.columns[cols]]

    crossValidationTestAndPlot(LogisticRegression(), "100-Best Features With Polynomials up to n^2",
                               poly3Data, labels, cvNum = 5, addAverage=True)

    

    plt.legend()
    plt.show()


def recursiveComparion(data, labels):
    estimator = LogisticRegression()
    rfecv = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(5),
              scoring='accuracy')
    rfecv.fit(data, labels)
    rfe = RFE(estimator=estimator, step=1)
    rfe.fit(data, labels)

    print("Optimal number of features : %d" % rfecv.n_features_)
    print("Features selected: ", rfecv.support_ )
    print("Feature ranking: ", rfecv.ranking_ )

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()

    featuresWithRankings = zip(data.columns.values.tolist(), rfecv.ranking_)
    for column, rank in featuresWithRankings:
        if rank == 1:
            print (column + ","+str(rank))
    #plt.imshow(featuresWithRankings, cmap='hot', interpolation='nearest')
    #plt.show()


    plotROCCurveBase()

    crossValidationTestAndPlot(LogisticRegression(), "Full Feature Set",
                               data, labels, cvNum = 5, addAverage=True)
    crossValidationTestAndPlot(rfecv, "After recursive features selection",
                               data, labels, cvNum = 5, addAverage=True)

    plt.legend()
    plt.show()

def weightedComp():
    plotROCCurveBase()

    crossValidationTestAndPlot(LogisticRegression(), "Unweighted",
                               data, labels, cvNum = 5, addAverage=True)
    crossValidationTestAndPlot(LogisticRegression(class_weight='balanced'),
                                                  "Weighted - Balanced",
                               data, labels, cvNum = 5, addAverage=True)

    plt.legend()
    plt.show()
    


    
    
def main():
    dataToDrop = ["YEAR", "TAIL_NUM", "FL_NUM",
         "DEP_DELAY","DEP_DEL15", "DEP_DELAY_GROUP",
                  "TAXI_OUT", "WHEELS_OFF", "WHEELS_ON", "TAXI_IN",
                   "ARR_TIME", "ARR_DELAY", "ARR_DEL15",
                  "ARR_DELAY_GROUP",  "DEP_TIME"]
    #"CRS_DEP_TIME", "CRS_ELAPSED_TIME", "CRS_ARR_TIME",
    data_full = pd.read_csv(DATA)
    data= data_full #.sample(n=1000)
    

    labels = []
    for index, row in data.iterrows():
    #for index, row in islice(data.iterrows(), LIMIT):
         if float(row["DEP_DELAY"]) > 0:
             labels.append(1)
         else:
             labels.append(0)
    data = data.drop(dataToDrop, 1)
    sum = 0
    for l in labels:
         sum += l
    #print(labels)

    print (sum/len(labels))
    labels = np.asarray(labels)

    categorical = data.select_dtypes(exclude='number')
    numerical = data.select_dtypes(include='number')

    print("Encoding")
    encoded = encode(categorical)
    print("Finished Encoding")

    data = pd.concat([numerical, encoded], axis=1)
    #print(data.head())

    #crossValidation(data, labels)
    #featureSelection(data, labels)
    

    #featureSelection(data, labels)
    # Create and fit selector
    selector = SelectKBest(k=100)
    selector.fit(data, labels)
    # Get columns to keep
    cols = selector.get_support()
    print(len(cols))
    # Create new dataframe with only desired columns, or overwrite existing
    data = data[data.columns[cols]]

    F, pval = f_classif(data, labels)
    headers = data.columns.values.tolist()

    #summary = pd.DataFrame([headers, F.tolist(), pval.tolist()])
    #summary.to_csv("summary.csv", mode='a')


    #print("F: ", F)
    #print("pval: ", pval)
    #notStatSig_old = ["ORIGIN_SJU", "DEST_CID", "DEST_JFK", "ORIGIN_IAD", "UNIQUE_CARRIER_AA", "DEST_TUL", "ORIGIN_PNS", "ORIGIN_SHV", "ORIGIN_SBP", "ORIGIN_MRY", "ORIGIN_SAF", "DEST_MFE", "DEST_MBS", "DEST_CMI", "DEST_PDX", "ORIGIN_PVD", "DEST_PSC", "ORIGIN_SGF", "ORIGIN_MTJ", "ORIGIN_FNT", "DEST_ANC", "ORIGIN_CLE", "ORIGIN_RDM", "DEST_RST", "ORIGIN_HPN", "UNIQUE_CARRIER_HA", "ORIGIN_DAY", "ORIGIN_SAN", "ORIGIN_SDF", "ORIGIN_MYR", "ORIGIN_ORD", "DEST_EYW", "UNIQUE_CARRIER_QX", "ORIGIN_ATW", "ORIGIN_BZN", "DEST_PIA", "UNIQUE_CARRIER_YV", "DEST_DSM", "DEST_PBI"]
    notStatSig = ["ORIGIN_IAD","UNIQUE_CARRIER_AA","DEST_TUL","ORIGIN_PNS","ORIGIN_SHV","ORIGIN_SBP","ORIGIN_MRY","ORIGIN_SAF","DEST_MFE","DEST_MBS","DEST_CMI","DEST_PDX","ORIGIN_PVD","DEST_PSC","ORIGIN_SGF","ORIGIN_MTJ","ORIGIN_FNT","DEST_ANC","ORIGIN_CLE","ORIGIN_RDM","DEST_RST","ORIGIN_HPN","UNIQUE_CARRIER_HA","ORIGIN_DAY","ORIGIN_SAN","ORIGIN_SDF","ORIGIN_MYR","ORIGIN_ORD","DEST_EYW","UNIQUE_CARRIER_QX","ORIGIN_ATW","ORIGIN_BZN","DEST_PIA","UNIQUE_CARRIER_YV","DEST_DSM","DEST_PBI"]
    
    data = data.drop(notStatSig, 1)
    headers = data.columns.values.tolist()
    #headers.insert(0, 'Intercept')
    
    #recursiveComparion(data, labels)
    lr = LogisticRegression(class_weight='balanced').fit(data, labels)
    coef = pd.DataFrame(headers, lr.coef_.tolist())
    coef.to_csv("coef.csv")
    
    
main()
