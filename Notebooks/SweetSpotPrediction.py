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

def plot_map(df, map_column, dtype='numeric', figax=None):
    """
    Pull single map column from dataframe and plot
    Arguments:
        df - dataframe, with columns XPos, YPos, and maps
        map_column - str, column name to map
        dtype - str, data type of map, either 'numeric' for contour plot or 'cat' for bitmap
        figax - tuple of matplotlib (fig, axes) to plot in, if none will generate own plot
    """
    
    # Reshape dataframe with desired column values, get geometry
    df_plot = df.pivot(index='YPos', columns='XPos', values=map_column)
    
    if figax is None:
        fig, ax = plt.subplots(figsize=(12,8))
    else:
        fig, ax = figax
        
    # Select colour map
    if 'Uncertainty' in map_column:
        cm = 'summer'
    else:
        cm = 'plasma'
    
    # Prepare data and plot for continuous numerical input
    if dtype == 'numeric':
        
        X,Y = np.meshgrid(df_plot.columns.values, df_plot.index.values)
        
        cf = ax.contourf(X, Y, df_plot, cmap=cm)
        c = ax.contour(X, Y, df_plot, colors='black')
        
    elif dtype == 'cat':
        
        xmin, xmax = df_plot.columns.values[0], df_plot.columns.values[-1]
        ymin, ymax = df_plot.index.values[0], df_plot.index.values[-1]
        dx = (xmax - xmin) / (len(df_plot.columns) - 1)
        dy = (ymax - ymin) / (len(df_plot.index) - 1)
        
        cf = ax.imshow(df_plot, cmap=cm, origin='lower', extent=(xmin-0.5*dx, xmax+0.5*dx, ymin-0.5*dy, ymax+0.5*dy))
        
    else:
        raise ValueError("dtype argument must be either 'numeric' or 'cat'")
    
    # Adjust plot settings
    ax.set_aspect('equal', 'box')
    fig.colorbar(cf, ax=ax, shrink=0.8);
    ax.set_xlabel('x [m]', fontsize=14)
    ax.set_ylabel('y [m]', fontsize=14)
    ax.set_title(map_column, fontsize=18)
    fig.tight_layout()
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

def random_perturb(x, y, z, mean_pert=1, variability=0.1):
    """
    Function to apply random perturbation to z.
    
    Arguments:
        x - pandas series of x values
        y - pandas series of y values
        z - pandas series of z values
        mean_pert - overall multiplier, if None will select randomly between 0.8 and 1.2
        variability - max perturbation
        
    Returns:
        zscaled - pandas series of scaled z values
    """
    
    # Compute mean_pert if necessary
    if mean_pert is None:
        mean_pert = 0.8 + 0.4 * np.random.random()
    
    # Compute random phases
    xphase = 2 * np.pi * np.random.random()
    yphase = 2 * np.pi * np.random.random()
    
    # Compute frequencies (1/2 wavelength for extent of map)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xfreq = np.pi / (xmax - xmin)
    yfreq = np.pi / (ymax - ymin)
    
    # Compute multiplier & scaled map
    multiplier = mean_pert + variability * np.sin(xfreq*(x-xmin) + xphase) * np.sin(yfreq*(y-ymin) + yphase)
    zscaled = multiplier * z
    
    return zscaled