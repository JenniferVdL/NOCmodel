import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools

from collections import OrderedDict

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.externals import joblib

# Algorithms wich will be evaluated.
models = [
    DecisionTreeClassifier(),
    GaussianNB(),
    KNeighborsClassifier(),
    LinearDiscriminantAnalysis(),
    LinearSVC(),
    LogisticRegression(),
    MLPClassifier(),
    RandomForestClassifier(),
    SVC(probability=True),
    GradientBoostingClassifier()
    ]

allFeatures = {}
results = open("partialCorrelationResults.txt", "r") # Change file name
results.readline()
x = 0
for lines in results:
    x+=1
    line = lines.rstrip().split("\t")
    Features = eval(line[2])
    del(Features[0])
    allFeatures[x] = Features

results.close()

#Function for creating plots. Predicted NOC vs True NOC.
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, size=20)
    plt.yticks(tick_marks, classes, size = 20)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 size = 20)
    #plt.tight_layout()
    plt.ylabel('True NOC', size = 20)
    plt.xlabel('Predicted NOC', size =20)

def gettingData(FEATURES):
    with open("FeaturesTest.txt") as json_file:
        FeaturesTest = json.load(json_file)
    with open("FeaturesTraining.txt") as json_file:
        FeaturesTraining = json.load(json_file)
    training_ordered_features = []
    training_labels = []
    for sample, items in FeaturesTraining.items():
        sampleList = []
        training_labels.append(int(items['NOC']))
        for locus, item in sorted(items["Locus"].items()):
            for feature, featValue in sorted(item.items()):
                if feature in FEATURES:
                    sampleList.append(float(featValue))
                else:
                    continue
        del items['Locus']
        del items['NOC']
        for feature in sorted(items):
            if feature in FEATURES:
                continue
            else:
                del items[feature]
        ordered = OrderedDict(sorted(items.items()))
        newSampleList = sampleList + ordered.values()
        training_ordered_features.append(map(float, newSampleList))
    test_labels = []
    test_ordered_features = []
    for sample, items in FeaturesTest.items():
        sampleList = []
        test_labels.append(int(items['NOC']))
        for locus, item in sorted(items["Locus"].items()):
            for feature, featValue in sorted(item.items()):
                if feature in FEATURES:
                    sampleList.append(float(featValue))
                else:
                    continue
        del items['Locus']
        del items['NOC']
        for feature in sorted(items):
            if feature in FEATURES:
                continue
            else:
                del items[feature]
        ordered = OrderedDict(sorted(items.items()))
        newSampleList = sampleList + ordered.values()
        test_ordered_features.append(map(float, newSampleList))
    training_features = np.array(training_ordered_features)
    test_features = np.array(test_ordered_features)
    scaler = StandardScaler()
    training_features=scaler.fit_transform(training_features)
    test_features=scaler.transform(test_features)
    joblib.dump(scaler, "Scaler.pkl")
    return training_features, training_labels, test_features, test_labels


params = {}
for model in models:
    modelName = str(model).split("(")[0]
    if modelName == "DummyClassifier":
        continue
    params[modelName] = {}

#LinearDiscriminantAnalysis
params["LinearDiscriminantAnalysis"] = {
    "params1" : {"solver" : ["svd", "lsqr"], "n_components" : range(11)+[None],
     "store_covariance": [True, False], "tol" : [0.0,0.1, 0.01, 0.001, 0.0001, 0.00001]},
    "params2" : {"solver" : ["lsqr"], "shrinkage" : [None, "auto"],
        "n_components" : [None, range(11)],
        "tol" : [0.0,0.1, 0.01, 0.001, 0.0001, 0.00001]}
    }
params["LinearDiscriminantAnalysis"]["params2"]["shrinkage"].extend(np.arange(0.0, 1.01, 0.01))

#LinearSVC
params["LinearSVC"] = {
    "params1" : {"penalty" : ["l2"], "loss" : ["hinge", "squared_hinge"], "tol" : [0.1,0.001,0.0001,0.00001],
        "C" : [1, 10, 100, 1000], "multi_class": ["ovr", "crammer_singer"],  "fit_intercept" : [False, True]},
    "params2" : {"penalty" : ["l2"], "loss" : ["squared_hinge"], "tol" : [0.1,0.001,0.0001,0.00001],
        "C" : [1, 10, 100, 1000], "multi_class": ["ovr", "crammer_singer"],  "fit_intercept" : [False, True], "dual":[False,True]},
    "params3" : {"penalty" : ["l1", "l2"], "loss" : ["squared_hinge"], "tol" : [0.1,0.001,0.0001,0.00001],
        "C" : [1, 10, 100, 1000], "multi_class": ["ovr", "crammer_singer"],  "fit_intercept" : [False, True], "dual":[False]}
    }

#KNeighborsClassifier
params["KNeighborsClassifier"] = {
    "params1" : {"n_neighbors" : [i for i in range(1,11)], "weights" : ["uniform", "distance"],
        "algorithm" : ["auto"], "leaf_size" : [i for i in range(20,41)],
        "p" : [1,2,3], "metric" : ["euclidean", "manhattan", "chebyshev", "minkowski"]}
    }

#SVC
params["SVC"] = {
    "params1": {"C" : [1, 10, 100, 1e3],
        "kernel" : ["linear", "rbf", "sigmoid"],
        "gamma" : [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
        "shrinking" : [False, True], "probability" : [True],
        "tol" : [0.1, 0.01, 0.001, 0.0001, 0.00001],
        "decision_function_shape" : ["ovo", "ovr"]}
    }

#DTC
params["DecisionTreeClassifier"] = {
    "params1" : {
        "criterion" : ["gini", "entropy"], "splitter" : ["best", "random"],
        "max_features" : ["auto", "sqrt", "log2", None],
        "presort" : [False, True],
        "max_depth" : range(1,11)+[None], "min_samples_split" : range(2,11),
        "max_leaf_nodes" : [None]}
    }


#MLP
params["MLPClassifier"] = {
    "params1" : {
        "hidden_layer_sizes": [(100,), (10,10)],
        "activation" : ["identity", "logistic", "tanh", "relu"],
        "solver" : ["lbfgs"], "alpha" : [0.1, 0.01, 0.001, 0.0001, 0.00001],
        "max_iter" : [10000]}
    }


#LogisticRegression
params["LogisticRegression"] = {
    "params1" : {"penalty" : ["l2"], "dual" : [False],
        "tol" : [0.1, 0.01, 0.001, 0.0001, 0.00001], "C" : [1, 10, 100, 1e3],
        "fit_intercept" : [False, True],
        "solver" : ["newton-cg", "lbfgs", "sag"], "max_iter" : [10000],
        "multi_class" : ["ovr", "multinomial"]},
    "params3" : {"penalty" : ["l1"],
        "tol" : [0.1, 0.01, 0.001, 0.0001, 0.00001], "C" : [1, 10, 100, 1e3],
        "fit_intercept" : [False, True],
        "solver" : ["saga"], "max_iter" : [10000],
        "multi_class" : ["ovr", "multinomial"]},
    }

#GaussianNB
params["GaussianNB"] = {
    "params1" : {}
    }

#RandomForestClassifier
params["RandomForestClassifier"] = {
    "params1" : {"n_estimators": [1,10,20], "criterion":["gini", "entropy"],
        "max_depth" : range(1,6)+[None], "min_samples_split" : range(2,6), "min_samples_leaf" : [0.5,1],
        "min_weight_fraction_leaf":[0.0], "max_features" : ["auto", "sqrt", "log2", None],
        "max_leaf_nodes" : range(10,16) + [None],
        "min_impurity_decrease":[0.0], "min_impurity_split":[None],
        "bootstrap":[True, False], "oob_score":[False]
        }
    }

#GradientBoostingClassifier
params["GradientBoostingClassifier"] = {
    "params1" : {"loss": ["deviance"], "criterion":["friedman_mse", "mse", "mae"],
        "max_depth" : range(1,6)+[None], "min_samples_split" : range(2,6), "min_samples_leaf" : [0.5,1],
        "min_impurity_decrease":[0.0], "min_impurity_split":[None],
        "learning_rate":[0.1], "n_estimators":[100, 1000]
        }
    }


results = open("NoGridResults.txt", "w")
gridResults = open("GridResults.txt", "w")
gridResults.write("Model\tTraining\tTest\tNrFeats\tParameters\n")
results.write("Model\tTraining\tTest\tNrFeats\n")
for model in models:
    modelName = str(model).split("(")[0]
    for nr in range(1,51):
        newFeatures = allFeatures[nr]
        training_features, training_labels, test_features, test_labels = gettingData(newFeatures)
        model.fit(training_features, training_labels)
        joblib.dump(model, modelName+"_"+str(nr)+".pkl")
        accTraining = accuracy_score(training_labels, model.predict(training_features))
        accTest = accuracy_score(test_labels, model.predict(test_features))
        # corr = np.corrcoef(model.predict(test_features), test_labels)[0][1]
        results.write("{0}\t{1}\t{2}\t{3}\n".format(modelName,accTraining,accTest, nr))
        # # Compute confusion matrix
        cnf_matrix = confusion_matrix(test_labels, model.predict(test_features))
        np.set_printoptions(precision=2)
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=range(1,6),
                              title=modelName)
        plt.savefig(modelName+"_"+str(nr)+"_Test.jpg")
        cnf_matrix = confusion_matrix(training_labels, model.predict(training_features))
        np.set_printoptions(precision=2)
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=range(1,6),
                              title=modelName)
        plt.savefig(modelName+"_"+str(nr)+"_Train.jpg")

        NOC_Labels = ["1", "2", "3", "4", "5"]
        report = open(modelName+"_"+str(nr)+"Report.txt", "w")
        classReport = classification_report(test_labels, model.predict(test_features), NOC_Labels, NOC_Labels)
        report.write(classReport)
        report.close()

        #GridSearch
        clf = GridSearchCV(estimator = model,
            param_grid = params[modelName].values(), n_jobs = 3)
        clf.fit(training_features, training_labels)
        clf.best_estimator_.fit(training_features, training_labels)
        joblib.dump(clf.best_estimator_, modelName+"_"+str(nr)+"GridSearch.pkl")
        accTraining = accuracy_score(training_labels, clf.best_estimator_.predict(training_features))
        accTest = accuracy_score(test_labels, clf.best_estimator_.predict(test_features))
        # corr = np.corrcoef(clf.best_estimator_.predict(test_features), test_labels)[0][1]
        gridResults.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(modelName,accTraining,accTest, nr, str(clf.best_estimator_).replace("\n", "")))
        # # Compute confusion matrix
        cnf_matrix = confusion_matrix(test_labels, clf.best_estimator_.predict(test_features))
        np.set_printoptions(precision=2)
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=range(1,6),
                              title=modelName)
        plt.savefig(modelName+"_"+str(nr)+"_GridSearchTest.jpg")
        cnf_matrix = confusion_matrix(training_labels, clf.best_estimator_.predict(training_features))
        np.set_printoptions(precision=2)
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=range(1,6),
                              title=modelName)
        plt.savefig(modelName+"_"+str(nr)+"_GridsearchTrain.jpg")

        NOC_Labels = ["1", "2", "3", "4", "5"]
        report = open(modelName+"_"+str(nr)+"ReportgridResults.txt", "w")
        classReport = classification_report(test_labels,    els)
        report.write(classReport)
        report.close()
        plt.close("all")



results.close()
gridResults.close()

