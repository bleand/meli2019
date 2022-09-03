import os
import time
import joblib
import operator
import csv

import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

matplotlib.use('pdf')
plt.style.use('ggplot')


def plot_history(history, pref=None):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    if pref is not None:
        plt.savefig(str(pref) + '_foo.png')
    else:
        plt.savefig('foo.png')

def get_oldest_file(files, _invert=False):
    """ Find and return the oldest file of input file names.
    Only one wins tie. Values based on time distance from present.
    Use of `_invert` inverts logic to make this a youngest routine,
    to be used more clearly via `get_youngest_file`.
    """
    gt = operator.lt if _invert else operator.gt
    # Check for empty list.
    if not files:
        return None
    # Raw epoch distance.
    now = time.time()
    # Select first as arbitrary sentinel file, storing name and age.
    oldest = files[0], now - os.path.getctime(files[0])
    # Iterate over all remaining files.
    for f in files[1:]:
        age = now - os.path.getctime(f)
        if gt(age, oldest[1]):
            # Set new oldest.
            oldest = f, age
    # Return just the name of oldest file.
    return oldest[0]

def get_youngest_file(files):
    return get_oldest_file(files, _invert=True)

def output_csv(y_true,y_pred,  orig_test, pad_test, language='spanish', path='', bacc = 0):
    levels = joblib.load(language + '/levels')
    csv_file = open(language+'/Output_CNN_' + str(language) + '.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    writer.writerow(
        ['Original','Padded','Human', 'Pred', 'Prob', 'BACC'])
    ix = 0
    cat_true = []
    cat_pred = []
    first = 0
    for y,h,o,p in zip(y_true,y_pred,orig_test, pad_test):
        cat_true.append(levels[np.argmax(y)])
        cat_pred.append(levels[np.argmax(h)])
        if first == 0:
            writer.writerow([o,p,levels[np.argmax(y)], levels[np.argmax(h)],h[np.argmax(h)],bacc])
            first = 1
        else:
            writer.writerow([o,p,levels[np.argmax(y)], levels[np.argmax(h)], h[np.argmax(h)]])
    report = classification_report(cat_true, cat_pred, output_dict=True)
    df = pd.DataFrame(report)
    df.to_csv(language+'/Report.csv')

    csv_file.close()
