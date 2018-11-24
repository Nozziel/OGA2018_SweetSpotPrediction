import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

def ReportMetrics(model, X_train, X_test, y_train, y_test, y_pred_test):
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn import metrics

    weights_train = compute_sample_weight(class_weight='balanced', y=y_train)
    weights_test = compute_sample_weight(class_weight='balanced', y=y_test)

    print("Weighted accuracy on training set: {:.3f}".format(model.score(X_train, y_train, sample_weight=weights_train)))
    print("Weighted accuracy on test set: {:.3f}".format(model.score(X_test, y_test,weights_test)))

    print("")

    # Model Precision: number of positive predictions divided by the total number of positive class values predicted.
    print("Precision: {:.3f}".format(metrics.precision_score(y_test, y_pred_test)))

    # Model Recall: the number of positive predictions divided by the number of positive class values in the data
    print("Recall: {:.3f}".format(metrics.recall_score(y_test, y_pred_test)))

    # Model Recall: 2*((precision*recall)/(precision+recall)).
    print("F1: {:.3f}".format(metrics.f1_score(y_test, y_pred_test)))

    return

def plot_feature_importances(model,features):
    fig, ax = plt.subplots(figsize=(20, 10))
    n_features = len(features)
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    return