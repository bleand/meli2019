import pandas as pd
from data_helpers import clean_str, load_data, indices_to_one_hot
from utils import plot_history, output_csv
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import copy
from sklearn.utils import class_weight
from sklearn.metrics import balanced_accuracy_score
import gc
from models import cnn_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from statics import *
from tqdm.auto import tqdm
import logging

tqdm.pandas()

logger = logging.getLogger(__name__)

DATA_PATH = './../data/'
RAW_TRAIN = os.path.join(DATA_PATH, 'train.csv.gz')
DATASET_FILENAME = os.path.join(DATA_PATH, 'clean20M.csv')


def pipeline(args):

    if args.clean_tset:

        # clean-tset means the code starts with raw data
        logger.info("Starting with clean dataset.")
        training_set = pd.read_csv(RAW_TRAIN, compression='gzip')

        if args.sample:
            # In case we want to test the code, we can take a small sample of the data
            logger.info("Extracting a sample from dataset.")
            training_set = training_set.sample(200000)
            logger.debug(f"New training set shape is {training_set.shape}")

        logger.info("Cleaning titles.")
        training_set['clean_title'] = training_set.progress_apply(lambda x: clean_str(x['title']), axis=1)
        training_set.to_csv(DATASET_FILENAME)

    else:

        # If clean-tset is not present, we load an existing clean dataset
        logger.info(f"Loading saved dataset. Filename {DATASET_FILENAME}")
        training_set = pd.read_csv(DATASET_FILENAME)

    for language in ['spanish', 'portuguese']:
        logger.info(f"Starting with {language}.")
        prefix = language[0:3]
        path = os.path.join(DATA_PATH, language)
        logger.info(f"Language path: {path}.")
        Path(path).mkdir(parents=True, exist_ok=True)

        if args.create_tset:
            logger.info(f"Starting to pre-process dataset.")
            if args.sample and not args.clean_tset:
                logger.info("Extracting a sample from dataset.")
                training_set = training_set.head(200000)

            # Extracting only data of the current language
            is_lang = training_set['language'] == language
            training_set_lang = training_set[is_lang].copy()

            # Load and pre-process data
            df, len_sent = load_data(training_set_lang, path=path, reliable=False, lang=language)

            # If it's a sample, we need to remove categories with less samples than 2
            if args.sample:
                logger.info("Cleaning categories from sample.")
                df = df[~df.category.isin(
                    df.category.value_counts()[df.category.value_counts() < 2].index)]

            df = df.drop(columns=['clean_title'])

            logger.info("Train test split.")
            train_df, test_df = train_test_split(df, test_size=0.1, stratify=df['category'])

            logger.info(f"Dumping dataset into {path}.")
            joblib.dump(len_sent, os.path.join(path, f'len_sent_{prefix}.h5'))
            train_df.to_pickle(os.path.join(path, f'df_{prefix}.pkl'))
            test_df.to_pickle(os.path.join(path, f'df_{prefix}_test.pkl'))

        if args.train:

            logger.info(f"Training...")

            if not args.create_tset:
                logger.info(f"Loading dataset from {path}.")
                df = pd.read_pickle(os.path.join(path, f'df_{prefix}.pkl'))
                len_sent = joblib.load(os.path.join(path, f'len_sent_{prefix}.h5'))

            if args.calc_labels:
                logger.info(f"Binarizing labels.")
                nb_classes = len(np.unique(df['category']))
                labels, levels = pd.factorize(df['category'])
                joblib.dump(nb_classes, os.path.join(path, 'nb_classes'))
                joblib.dump(levels, os.path.join(path + 'levels'))
            else:
                nb_classes = joblib.load(os.path.join(path, 'nb_classes'))
                levels = joblib.load(os.path.join(path + 'levels'))

            logger.info("Train validation split.")
            train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['category'])

            logger.info("Extracting values from dataframes.")
            y = copy.deepcopy(train_df['category'].values)
            x = copy.deepcopy(train_df['input_data'].values)
            y_val_in = copy.deepcopy(val_df['category'].values)
            x_val_in = copy.deepcopy(val_df['input_data'].values)

            logger.debug("Calculating class weight.")
            y_w = [np.where(levels == i)[0][0] for i in y]
            class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                              classes=np.unique(y_w),
                                                              y=y_w)
            class_weights = dict(zip(np.unique(y_w), class_weights))

            # In order to save memory, unnecessary objects are dropped
            del train_df
            del val_df
            del df
            gc.collect()

            logger.debug("Encoding validation labels.")
            y_val = [np.where(levels == i)[0][0] for i in y_val_in]
            y_val = indices_to_one_hot(y_val, nb_classes)

            logger.debug("Stacking validation features.")
            x_val = np.stack(x_val_in)

            logger.info("Instantiating model.")
            model_path = os.path.join(path, f'my_model_{language}.h5')
            if args.fresh_start:
                logger.info(f"Fresh start saved to: {model_path}.")
                output_shape = y_val.shape[1]
                model = cnn_model(len_sent, output_shape, path=path)
                adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
                model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
                model.save(model_path)
            else:
                logger.info(f"Loading model from: {model_path}.")
                model = load_model(model_path)
                adam = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
                model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

            if args.multi_gpu:
                logger.info(f"Transforming into a multi GPU model.")
                from tensorflow.keras.utils import multi_gpu_model
                adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
                parallel_model = multi_gpu_model(model, gpus=2)
                parallel_model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                                       optimizer=adam)

            logger.info(f"Model summary")
            logger.info(model.summary())

            checkpoint = ModelCheckpoint(os.path.join(path, 'weights.hdf5'), monitor='val_accuracy', verbose=2,
                                         save_best_only=True, mode='auto')
            early_stopping = EarlyStopping(monitor='val_loss', patience=6, verbose=1)
            reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.33, patience=2, verbose=1)

            batch = 10000000
            slices = (len(y) // batch + 1) if (len(y) % batch) > 0 else len(y) // batch
            logger.info(f"Model is splitted on {slices} slices of max {batch} size")

            for i in range(slices):

                print("Slice", i, "of", slices)
                logger.info(f"Slice {i} of {slices}")

                y_fold = y[i * batch:(i + 1) * batch]
                x_fold = np.stack(x[i * batch:(i + 1) * batch])

                y_fold = [np.where(levels == i)[0][0] for i in y_fold]
                y_fold = indices_to_one_hot(y_fold, nb_classes)

                if args.multi_gpu:
                    # Parallel model implementation
                    history = parallel_model.fit(x_fold, y_fold,
                                                 batch_size=batch_size, epochs=nb_epoch, verbose=1,
                                                 callbacks=[checkpoint, early_stopping, reduceLR],
                                                 validation_data=(x_val, y_val),
                                                 class_weight=class_weights)
                else:
                    history = model.fit(x_fold, y_fold,
                                        batch_size=batch_size, epochs=nb_epoch, verbose=1,
                                        callbacks=[checkpoint, early_stopping, reduceLR],
                                        validation_data=(x_val, y_val),
                                        class_weight=class_weights)

                logger.info(f"Plotting model history")
                plot_history(history, pref=language + str(i), path=path)

                logger.info(f"Saving final model into {model_path}")
                weight_file = os.path.join(path, 'weights.hdf5')
                if args.multi_gpu:
                    # In order to have a parallel model usable in a single-gpu configuration
                    # weights from trained parallel model need to be copied into the
                    # non-parallel version and saved.
                    parallel_model.load_weights(weight_file)
                    model.set_weights(parallel_model.get_weights())
                    model.save(model_path)
                else:
                    model.load_weights(weight_file)
                    model.save(model_path)

            if not args.create_tset:
                test_set_path = os.path.join(path, f'df_{prefix}_test.pkl')
                logger.info(f"Loading test set from: {test_set_path}")
                test_df = pd.read_pickle(test_set_path)

            logger.info(f"Generating test predictions")
            xt = np.stack(test_df['input_data'])
            outputs = model.predict(xt)

            logger.info(f"Calculating metrics.")
            yt = [np.where(levels == i)[0][0] for i in test_df['category']]
            yt = indices_to_one_hot(yt, nb_classes)
            bacc = balanced_accuracy_score(np.argmax(yt, axis=1), np.argmax(outputs, axis=1))

            logger.info(f"Test BACC: {bacc}")

            output_csv(yt, outputs, test_df['title'].values, test_df['sentences_padded'].values, language=language,
                       path=path,
                       bacc=bacc)

            # In order to save memory, unnecessary objects are dropped
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
            logger.info(f"Running evaluation...")

            test_set_path = os.path.join(path, f'df_{prefix}_test.pkl')
            logger.info(f"Loading test set from: {test_set_path}")
            test_df = pd.read_pickle(test_set_path)

            model_path = os.path.join(path, f'my_model_{language}.h5')
            logger.info(f"Loading model from: {model_path} - and label encoding from: {path}")
            model = load_model(model_path)
            nb_classes = joblib.load(os.path.join(path, 'nb_classes'))
            levels = joblib.load(os.path.join(path + 'levels'))

            logger.info(f"Generating test predictions")
            xt = np.stack(test_df['input_data'])
            outputs = model.predict(xt)

            logger.info(f"Calculating metrics.")
            yt = [np.where(levels == i)[0][0] for i in test_df['category']]
            yt = indices_to_one_hot(yt, nb_classes)
            bacc = balanced_accuracy_score(np.argmax(yt, axis=1), np.argmax(outputs, axis=1))

            logger.info(f"Test BACC: {bacc}")

            output_csv(yt, outputs, test_df['title'].values, test_df['sentences_padded'].values, language=language,
                       path= path,
                       bacc=bacc)
