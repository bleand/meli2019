import pandas as pd
import os
import matplotlib.pyplot as plt
from data_helpers import load_data, clean_str, indices_to_one_hot
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
from models import cnn_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from utils import plot_history, output_csv
from statics import *
import joblib
from sklearn.metrics import balanced_accuracy_score
from tensorflow.keras.models import load_model
import gc
import copy
from pathlib import Path
from sklearn.utils import class_weight

tqdm.pandas()
plt.style.use('ggplot')

# PARAMETERS

multi_gpu = False  # Initially, this code was run on a multi-gpu environment
SAMPLE = True  # Run code on a subset of the data. Useful to test functionality
clean_tset = True  # Run-preprocessing again. Useful to avoid re-running pre-processing when that stays the same
create_tset = True  #
train = True
calc_labels = True
fresh_start = True

if clean_tset:
    tset = pd.read_csv('./../data/train.csv.gz', compression='gzip')

    if SAMPLE:
        tset = tset.head(200000)

    def clean_df(row):
        return clean_str(row)

    tset['clean_title'] = tset.progress_apply(lambda x: clean_df(x['title']), axis=1)
    tset.to_csv('./../data/clean20M.csv')

elif create_tset:
    print('loading clean20M')
    tset = pd.read_csv('data/clean20M.csv')

for lang in ['spanish', 'portuguese']:

    suff = lang[0:3]
    path = os.path.join('./../data/', lang)
    print(path)
    Path(path).mkdir(parents=True, exist_ok=True)

    if create_tset:
        if SAMPLE and not clean_tset:
            tset = tset.head(200000)
        print('Creating tset for', lang)
        is_lang = tset['language'] == lang
        tset_lang = tset[is_lang]

        print(tset_lang.head())
        df, len_sent = load_data(tset_lang, path=path, reliable=False, lang=lang)

        if SAMPLE:
            for item in df.groupby('category')['title'].nunique().iteritems():
                if item[1] < 10:
                    df = df[tset.category != item[0]]

        print('Dropping column')
        df = df.drop(columns=['clean_title'])

        print('Splitting Training set')
        train_df, test_df = train_test_split(df, test_size=0.1, stratify=df['category'])

        print('Saving training set')
        joblib.dump(len_sent, path + '/len_sent_' + suff + '.h5')
        train_df.to_pickle(path + '/df_' + suff + '.pkl')
        test_df.to_pickle(path + '/df_' + suff + '_test.pkl')

    if train:

        print('reading1')
        df = pd.read_pickle(path + '/df_' + suff + '.pkl')
        len_sent = joblib.load(path + '/len_sent_' + suff + '.h5')

        if calc_labels:
            nb_classes = len(np.unique(df['category']))
            labels, levels = pd.factorize(df['category'])
            joblib.dump(nb_classes, path + '/nb_classes')
            joblib.dump(levels, path + '/levels')
        else:
            nb_classes = joblib.load(path + '/nb_classes')
            levels = joblib.load(path + '/levels')

        train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['category'])
        y = copy.deepcopy(train_df['category'].values)
        x = copy.deepcopy(train_df['input_data'].values)
        y_val_in = copy.deepcopy(val_df['category'].values)
        x_val_in = copy.deepcopy(val_df['input_data'].values)

        y_w = [np.where(levels == i)[0][0] for i in y]

        class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                          classes=np.unique(y_w),
                                                          y=y_w)
        class_weights = dict(zip(np.unique(y_w), class_weights))

        print('deleting1')

        del train_df
        del val_df
        del df

        gc.collect()

        y_val = [np.where(levels == i)[0][0] for i in y_val_in]
        y_val = indices_to_one_hot(y_val, nb_classes)

        print('stack1')
        x_val = np.stack(x_val_in)

        print("Instantiating model...")
        if fresh_start:
            output_shape = y_val.shape[1]
            model = cnn_model(len_sent, output_shape, path=path)
            adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
            model.save(os.path.join(path, 'my_model_' + lang + '.h5'))
        else:
            model = load_model(os.path.join(path, 'my_model_' + lang + '.h5'))
            adam = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

        print(model.summary())
        checkpoint = ModelCheckpoint(os.path.join(path, 'weights.hdf5'), monitor='val_acc', verbose=2,
                                     save_best_only=True, mode='auto')
        early_stopping = EarlyStopping(monitor='val_loss', patience=6, verbose=1)
        reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.33, patience=2, verbose=1)

        batch = 10000000
        slices = (len(y) // batch + 1) if (len(y) % batch) > 0 else len(y) // batch
        for i in range(slices):
            if i <= -1:
                continue
            print("Slice", i, "of", slices)
            print('fold1')
            y_fold = y[i * batch:(i + 1) * batch]
            x_fold = np.stack(x[i * batch:(i + 1) * batch])

            y_fold = [np.where(levels == i)[0][0] for i in y_fold]
            y_fold = indices_to_one_hot(y_fold, nb_classes)

            if multi_gpu:
                history = parallel_model.fit(x_fold, y_fold, batch_size=batch_size, epochs=nb_epoch, verbose=1,
                                             callbacks=[checkpoint, early_stopping, reduceLR],
                                             validation_data=(x_val, y_val), class_weight='auto')

            else:

                history = model.fit(x_fold, y_fold, batch_size=batch_size, epochs=nb_epoch, verbose=1,
                                    callbacks=[checkpoint, early_stopping, reduceLR], validation_data=(x_val, y_val),
                                    class_weight=class_weights)
            plot_history(history, pref=lang + str(i))

            # files = glob.glob(path+'/weights.*.hdf5')
            # weight_file = get_youngest_file(files)
            weight_file = path + '/weights.hdf5'

            if multi_gpu:
                parallel_model.load_weights(weight_file)
                model.set_weights(parallel_model.get_weights())
                model.save(path + '/my_model_' + lang + '.h5')
            else:
                model.load_weights(weight_file)
                model.save(path + '/my_model_' + lang + '.h5')

        df_test = pd.read_pickle(lang + '/df_' + suff + '_test.pkl')
        xt = np.stack(df_test['input_data'])
        outputs = model.predict(xt)

        labels = []
        for i in df_test['category']:
            labels.append(np.where(levels == i)[0][0])
        labels = np.array(labels)
        print('labels processed')
        yt = indices_to_one_hot(labels, nb_classes)
        bacc = balanced_accuracy_score(np.argmax(yt, axis=1), np.argmax(outputs, axis=1))

        output_csv(yt, outputs, df_test['title'].values, df_test['sentences_padded'].values, language=lang,
                   bacc=bacc)
        print(bacc)

        del y
        del x
        del y_val_in
        del x_val_in
        del y_fold
        del x_fold
        del x_val
        del y_val

        gc.collect()

    else:
        df_test = pd.read_pickle(path + '/df_' + suff + '_test.pkl')
        print('predicting')
        model = load_model(path + '/my_model_' + lang + '.h5')
        levels = joblib.load(path + '/levels')
        nb_classes = joblib.load(path + '/nb_classes')
        print('model loaded')
        labels = []
        for i in df_test['category']:
            labels.append(np.where(levels == i)[0][0])
        labels = np.array(labels)
        print('labels processed')
        yt = indices_to_one_hot(labels, nb_classes)
        print('onehot done')
        xt = np.stack(df_test['input_data'])
        outputs = model.predict(xt)
        print('predicted')
        bacc = balanced_accuracy_score(np.argmax(yt, axis=1), np.argmax(outputs, axis=1))
        print(bacc)
        output_csv(yt, outputs, df_test['title'].values, df_test['sentences_padded'].values, language=lang, bacc=bacc)