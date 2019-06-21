import os
import itertools
import numpy as np
import matplotlib as mpl
#mpl.use('pdf')
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#########################################################
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
                          #cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    mpl.rcParams['axes.linewidth'] = 0.1
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=8)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes,fontsize=5, rotation=45)
    plt.yticks(tick_marks, classes,fontsize=5)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", fontsize=4,
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Pressed Keys', fontsize=8)
    plt.xlabel('Predicted Pressed Keys', fontsize=8)

    """
    plt.ylabel('True Fingers', fontsize=8)
    plt.xlabel('Predicted Fingers', fontsize=8)
    """
    plt.tight_layout()
#########################################################

print ('loading cm..')
cm = np.load('cm_hanon_0.95.npy')
print cm.dtype
print cm.shape
cm = cm.astype(int)
#print cm
print cm.dtype

"""
# finger identification
cm_finger = cm[7:17,7:17]
print cm_finger.shape
#class_names = [str(i) for i in range(0, 10)]

finger_names = ['thumb(r)', 'index(r)', 'middle(r)',' ring(r)', 'little(r)',
                'thumb(l)', 'index(l)', 'middle(l)',' ring(l)', 'little(l)']

# plot confusion matrix
plot_confusion_matrix(cm_finger, classes=finger_names,
                      normalize=True,
                      title='Finger Identification on Hanon Excercises')
plt.savefig('confusion_matrix_finger_norm.pdf')
"""

# key detection
cm_key = cm[:7,:7]
print cm_key.shape
key_names = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

# plot confusion matrix
plot_confusion_matrix(cm_key, classes=key_names,
                      normalize=True,
                      title='Pressed Key Detection on Hanon Excercises')
plt.savefig('confusion_matrix_key_norm.pdf')
