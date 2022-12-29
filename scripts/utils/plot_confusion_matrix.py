import numpy as np
import glob
import json
import os 
import matplotlib.pyplot as plt
import numpy as np
import itertools
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

import numpy as np

root_add = "/home/mona/VideoMAE/dataset/somethingsomething/"
# path = r'/home/mona/VideoMAE/results/finetune_Allclass_BB(800)/*.txt'
# path = r'/home/mona/VideoMAE/results/finetune_Allclass_BB_VideoMAE_scratch(50)/*.txt'
path = r'/home/mona/VideoMAE/results/finetune_Allclass_from_coppola/*.txt'


files = glob.glob(path, recursive=True)
files = [file for file in files if 'log' not in file]


plot_name = files[0].split('/')[-2].split('.')[0]

label = []
pred = []
for file in files:
    f = open(file,'r')
    for i, line in enumerate (f):
        if i == 0 :
            continue
        else:
            prob = "".join(line.split(" ")[1:-3])[1:-1].split(",")
            prob = [float(p) for p in prob]
            prob = np.array(prob)
            max_index = np.argmax(prob)
            pred.append(max_index)
            GT = int(line.split(" ")[-3])
            label.append(GT)
            
        



f = open(os.path.join(root_add, 'labels','labels.json'))
labels = json.load(f)
class_names = list(labels.keys())
# for i in range (len(class_names)):
#     print (f"{i} : {class_names[i]}")



def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(80, 60))
    plt.title(title)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.1f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(f'{plot_name}.png', dpi=300)


cm = confusion_matrix(label, pred)

# print classes with the highest number of errors
errors = np.sum(cm, axis=1) - np.diag(cm)
error_percentage = errors / np.sum(cm, axis=1)
# sort the errors based on the percentage of errors
idx = np.argsort(error_percentage)
idx = idx[::-1]
for i in idx:
    print('class "%s": %d errors (%.2f%%) with %d samples with ID: %s' % (class_names[i], errors[i], 100.*errors[i]/np.sum(cm[i,:]), np.sum(cm[i,:]), i))

# ## analysis of the confusion matrix fo each class (Debugging)(3 most erroe : 109,47,12)
cm_norm = confusion_matrix(label, pred).astype('float') / confusion_matrix(label, pred).sum(axis=1)[:, np.newaxis]
for i in idx[0:10]:
    print('*'*50)
    print('class "%s": %d errors (%.2f%%) with ID: %s' % (class_names[i], errors[i], 100.*errors[i]/np.sum(cm[i,:]), i))
    cls = cm_norm[i, :] 
    cls_idx = np.argsort(cls)
    cls_idx = cls_idx[::-1]
    print('most confused classes:')
    for j in cls_idx[0:5]:
        if j != i:
            print('  class "%s": %d errors (%.2f%%) with ID: %s' % (class_names[j], errors[j], 100.*errors[j]/np.sum(cm[j,:]), j))

# # check single class
# i = 111
# print('class "%s": %d errors (%.2f%%) with ID: %s' % (class_names[i], errors[i], 100.*errors[i]/np.sum(cm[i,:]), i))
# cls = cm_norm[i, :] 
# cls_idx = np.argsort(cls)
# cls_idx = cls_idx[::-1]
# print('most confused classes:')
# for j in cls_idx[0:5]:
#     if j != i:
#         print('  class "%s": %d errors (%.2f%%) with ID: %s' % (class_names[j], errors[j], 100.*errors[j]/np.sum(cm[j,:]), j))


plot_confusion_matrix(cm = cm, 
                    normalize    = True,
                    target_names =  [str(num) for num in list(range(len(class_names)))],
                    title        = "Confusion Matrix")


# plot_confusion_matrix(cm = confusion_matrix(label,pred), 
#                     normalize    = True,
#                     target_names =  [str(num) for num in list(range(len(class_names)))],
#                     title        = "Confusion Matrix")