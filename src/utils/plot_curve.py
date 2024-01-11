from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc

import itertools
from itertools import cycle

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np
import torch


def plot_train_curve(epoch, train_list, valid_list, curve_type, title,
                     folder_path, fig_size=(12, 10), save_img=False):
  
  font_size = fig_size[0]

  fig, axes=plt.subplots(1, 1, figsize=fig_size)
  axes.yaxis.set_major_locator(MaxNLocator(12))
  axes.xaxis.set_major_locator(MaxNLocator(10))

  plt.plot(train_list, label='Train', linewidth=1.5)
  plt.plot(valid_list, label='Validation', linewidth=1.5)

  legend_properties = {'size':font_size+4, 'weight':'bold'}
  plt.legend(loc='best', shadow=True, prop=legend_properties)
  plt.grid(alpha=1)
  plt.xlim(0, epoch)
  plt.yticks(fontsize = font_size+4, fontweight='bold')
  plt.xticks(fontsize = font_size+4, fontweight='bold')
  plt.xlabel('Epochs', fontsize=font_size+8, fontweight='bold')
  plt.title(title, fontsize=font_size+10, fontweight='bold')

  if curve_type == 'Accuracy':
    plt.ylabel('Accuracy', fontsize=font_size+8, fontweight='bold')
    plt.ylim(-0.05, 1.05)
  elif curve_type == 'Loss':
    plt.ylabel('Loss', fontsize=font_size+8, fontweight='bold')
    plt.ylim(0, 5)
  else:
    raise ValueError('curve_type only is Accuracy or Loss')

  if save_img == True:
    img_name =  folder_path + curve_type + '_' + title + '.png'
    plt.savefig(img_name, bbox_inches='tight')

  plt.show()




def plot_confusion_matrix(cm, 
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=2) # set precision of floar, None = 8
    cm_original = cm.copy()
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("\nNormalized Confusion Matrix")
    else:
        print('\nConfusion Matrix, without normalization')

    plt.figure(figsize=(10, 10))
    
    CM_im = plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)              # Only Color map based on confusion_matrix
    plt.title(title, fontsize = 20, fontweight='bold')
    plt.colorbar(CM_im, fraction=0.0457, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize = 14, fontweight='bold')
    plt.yticks(tick_marks, classes, fontsize = 14, fontweight='bold')

    fmt = '.2f' if normalize else 'd'                                       # Define formate based on 'normalize = True or False'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):  # plot Number or Normalized Number in grid
        count = cm_original[i, j]
        percentage = format(cm[i, j], fmt)
        plt.text(j, i, f'{percentage}\n({count})',
                 horizontalalignment="center",
                 fontsize = 20,
                 fontweight='bold',
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize = 20, fontweight='bold')
    plt.xlabel('Predicted label', fontsize = 20, fontweight='bold')
    


  