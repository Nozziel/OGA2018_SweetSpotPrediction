import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi

def ReportMetrics(model, X_train, X_test, y_train, y_test, y_pred_test):
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn import metrics

    weights_train = compute_sample_weight(class_weight='balanced', y=y_train)
    weights_test = compute_sample_weight(class_weight='balanced', y=y_test)

  
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

def plot_map(df, map_column):
    """
    Pull single map column from dataframe and plot
    Arguments:
        df - dataframe, with columns XPos, YPos, and maps
        map_column - str, column name to map
    """
    
    # Reshape dataframe with desired column values, get geometry
    df_plot = df.pivot(index='YPos', columns='XPos', values=map_column)
    X,Y = np.meshgrid(df_plot.columns.values, df_plot.index.values)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12,8))
    cf = ax.contourf(X, Y, df_plot, cmap='plasma')
    c = ax.contour(X, Y, df_plot, colors='black')
    ax.set_aspect('equal', 'box')
    fig.tight_layout()
    fig.colorbar(cf, shrink=0.8);
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(map_column, fontsize=16)
    plt.show();
    return

def make_regular(df, name, spacing):
    """
    Takes dataframe for single map (XPos, YPos, name columns), interpolates it
    to regular rectangle with defined spacing
    """
    
    # Desired target grid skeleton 
    xmin, xmax = df.XPos.min(), df.XPos.max()
    ymin, ymax = df.YPos.min(), df.YPos.max()
    xtarget = np.arange(spacing*(xmin//spacing), spacing*(xmax//spacing + 2), spacing)
    ytarget = np.arange(spacing*(ymin//spacing), spacing*(ymax//spacing + 2), spacing)
    Xt, Yt = np.meshgrid(xtarget, ytarget)
    
    # Interpolate, and form into dataframe
    df_int = spi.griddata((df.XPos, df.YPos), df[name], (Xt, Yt), method='cubic')
    df_reg = pd.DataFrame({'XPos':Xt.flatten(), 'YPos':Yt.flatten(), name:df_int.flatten()})
    
    return df_reg