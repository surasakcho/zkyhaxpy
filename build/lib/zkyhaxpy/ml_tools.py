import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from matplotlib import rcParams

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          figsize=None,
                          save_as=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Tahoma']
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    
    cm_obs = confusion_matrix(y_true, y_pred)
    cm_pct = cm_obs.astype('float') / cm_obs.sum(axis=1)[:, np.newaxis] * 100
    
    # Only use the labels that appear in the data
    classes = np.array(classes)
    classes = classes[unique_labels(y_true, y_pred)]
    

    plt.figure(figsize=figsize)

    fig, ax = plt.subplots(figsize=figsize)
    if normalize==True:
        im = ax.imshow(cm_pct, interpolation='nearest', cmap=cmap)
        thresh = 50
        for i in range(cm_pct.shape[0]):
            for j in range(cm_pct.shape[1]):
                ax.text(j, i+.1, f'{cm_pct[i, j]:3.2f}%',
                        ha="center", 
                        va="center",
                        color="white" if cm_pct[i, j] > thresh else "black")
                        #color="black")
        for i in range(cm_pct.shape[0]):
            for j in range(cm_pct.shape[1]):
                ax.text(j, i-.1, f'({cm_obs[i, j]:,d})',
                        ha="center", 
                        va="center",
                        color="white" if cm_pct[i, j] > thresh else "black")
                        #color="black")
    else:
        im = ax.imshow(cm_obs, interpolation='nearest', cmap=cmap)
        thresh = cm_obs.max() / 2.
        for i in range(cm_obs.shape[0]):
            for j in range(cm_obs.shape[1]):
                ax.text(j, i, f'{cm_obs[i, j]}',
                        ha="center", va="center",
                        color="white" if cm_obs[i, j] > thresh else "black")

        
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm_obs.shape[1]),
           yticks=np.arange(cm_obs.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.    

        
    plt.xlim(-0.5,cm_obs.shape[1]-.5)
    plt.ylim(-0.5,cm_obs.shape[0]-.5)                    
    fig.tight_layout()
    if save_as != None:
        plt.savefig(save_as)
    plt.show()

    return cm_pct, cm_obs


def plot_roc(y_true, y_pred, classes,
                          
                          title=None,
                          cmap=plt.cm.Blues,
                          figsize=None,
                          save_as=None):
    #######INCOMPLETE######
    roc_filename = f'DecisionTree-{full_cluster_id}-{breed_code}-roc_curve.jpg'
    roc_path = os.path.join(cm_folder, roc_filename)
    title = f'ROC Curve - DecisionTree:{full_cluster_id}-{breed_nm}'    
    
    plt.title(title)
    list_legend = []
    for grp_2 in df_predict_result_grp_1.groupby('cloud_level'):
        cloud_level = grp_2[0]
        df_predict_result_grp_2 = grp_2[1]
        cloud_level_nm = dict_cloud_level[cloud_level]        
        fpr, tpr, thresholds = metrics.roc_curve(list(df_predict_result_grp_2['y_actual']), list(df_predict_result_grp_2['y_proba']))
        if cloud_level == 0:
            color = 'green'
        elif cloud_level == 1:
            color = 'blue'
        elif cloud_level == 2:
            color = 'orange'
        plt.plot(fpr, tpr, "-", color=color)
        list_legend.append(cloud_level_nm)
        
    plt.plot(fpr, fpr, "--")
    plt.xlabel("False Alarm")
    plt.ylabel("Detection")
    plt.legend(list_legend)    
    plt.grid()
    plt.savefig(roc_path)
    plt.show() 