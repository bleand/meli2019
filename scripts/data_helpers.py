import os
import json
import joblib
import re
import unidecode

from gensim.utils import SaveLoad
from gensim.models.phrases import Phraser, Phrases
from gensim.models.word2vec import Word2Vec

import numpy as np

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from tqdm.auto import tqdm
from statics import *


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data, dtype=np.int16).reshape(-1)
    return np.eye(nb_classes, dtype=np.int8)[targets]


def load_data(df, path='', reliable=False, lang=''):
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    # if os.path.isdir(path) == False:
    #     os.makedirs(path)

    print(path)
    df = load_data_and_labels(df, path=path, reliable=reliable, lang=lang)

    len_sent = 0
    for each in df['sentences_padded']:
        len_sent = len(each)
        break

    print('Building input data')
    df = build_input_data(df, path=path)

    return df, len_sent


def load_data_and_labels(df, path='', reliable=False, lang=''):
    print('Grabbing sentences...')
    sentences = df['clean_title'].astype(str)
    labels = df['category']

    x_text = sentences

    x_text = [s.split(" ") for s in x_text]

    print('Bigrams...')
    if reliable:
        bigram = SaveLoad.load(path + '/bigrams')
    else:
        bigram = Phraser(Phrases(x_text, min_count=100, threshold=20))
        bigram.save(path + '/bigrams')
    x_text_aux = x_text

    x_text = []
    print("Creating bi-grams")

    pbar = tqdm(total=len(x_text_aux))
    for text in x_text_aux:
        x_text.append(bigram[text])
        pbar.update()
    pbar.close()

    print('Removing stopwords...')

    x_text = remove_stopwords(x_text, lang=lang)

    """
    print('Stemming...')
    x_text = text_stemming(x_text, lang='spanish')
    """

    print('Padding sentences...')
    if reliable:
        len_sent = joblib.load(path + '/len_sent_' + lang[0:3] + '.h5')
        x_text = pad_sentences(x_text, len_sent=len_sent)
    else:
        x_text = pad_sentences(x_text)

    print("Word 2 Vec")
    if not reliable:
        model = Word2Vec(sentences=x_text, vector_size=EMBEDDING_DIM, sg=1, window=7, min_count=20, seed=42, workers=8)
        weights = model.wv.vectors

        np.save(open(path + '/embeddings.npz', 'wb'), weights)
        vocab = model.wv.key_to_index
        print(path)
        print(os.path.join(path, 'map.json'))
        with open(os.path.join(path, 'map.json'), 'w') as f:
            f.write(json.dumps(vocab))

    df['sentences_padded'] = x_text
    return df

def build_input_data(df, path=''):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    word2idx, idx2word = load_vocab(vocab_path='/map.json', path=path)
    df['input_data'] = df.apply(lambda x : np.array([word2idx[word] if word in word2idx.keys() else
                                                     word2idx["<PAD/>"] for word in x['sentences_padded']],
                                                    dtype=np.int32), axis=1)
    return df


def remove_stopwords(sentences, lang='english'):
    try:
        stpwrds = stopwords.words(lang)
    except Exception:
        stpwrds = stopwords.words('spanish')

    out_sentences = [[w for w in sentence if w not in stpwrds] for sentence in sentences]

    return out_sentences

def pad_sentences(sentences, padding_word="<PAD/>", len_sent = None):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    if len_sent is None:
        sequence_length = max(len(x) for x in sentences)
    else:
        sequence_length = len_sent
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        if num_padding >= 0:
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:sequence_length]
        padded_sentences.append(new_sentence)
    return padded_sentences

def load_vocab(vocab_path='/map.json', path=''):
    """
    Load word -> index and index -> word mappings
    :param vocab_path: where the word-index map is saved
    :return: word2idx, idx2word
    """

    with open(path+vocab_path, 'r') as f:
        data = json.loads(f.read())
    word2idx = data
    idx2word = dict([(v, k) for k, v in data.items()])
    return word2idx, idx2word


def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """

    string = unidecode.unidecode(string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r'(.)\1{3,}', r'\1\1', string)

    string = re.sub(r'\d{7,}', r' <longnumber> ', string)
    string = re.sub(r'\d{7,}', r' <longnumber> ', string)
    string = re.sub(r'(?<=[A-Za-z])[\d,]{10,}', r' <model> ', string)
    string = re.sub(r'[\d,]{10,}', r' <weird> ', string)

    # Spatial
    string = re.sub(r"((\d)+(\,|\.){0,1}(\d)*( ){0,1}((mts*)|(pulgadas*)|('')|"
                    "(polegadas*)|(m)|(mms*)|(cms*)|(metros*)|(mtrs*)|(centimetros*))+)"
                    "(?!(?!x)[A-Za-z])", " <smeasure> ", string, flags=re.I)
    string = re.sub(r"(mts)+( ){0,1}((\d)+(\,|\.){0,1}(\d)*)(?![A-Za-z])", " <smeasure> ", string, flags=re.I)
    string = re.sub(r"<smeasure> +[\/x] +<smeasure> +[\/x] +<smeasure>", " <smeasure> ", string, flags=re.I)
    string = re.sub(r"(<smeasure>|(\d)+(\,|\.)*(\d*)) +[\/x] "
                    "+(<smeasure>|(\d)+(\,|\.)*(\d*)) +[\/x] +<smeasure>", " <smeasure> ", string,
                    flags=re.I)
    string = re.sub(r"<smeasure> +[\/x] +<smeasure>", " <smeasure> ", string, flags=re.I)
    string = re.sub(r"(?<![\dA-Za-z])(<smeasure>|(\d)+(\,|\.)*(\d*)) *[\/x-] *<smeasure>",
                    " <smeasure> ", string, flags=re.I)
    string = re.sub(r"(?<![\dA-Za-z])((\d)+(\,|\.)*(\d*))x((\d)+(\,|\.)*(\d*))x +<smeasure>", " <smeasure> ",
                    string, flags=re.I)

    # Electrical
    string = re.sub(r"(?<![\dA-Za-z])((\d)+(\,|\.)*(\d*)) *amperes", " <emeasure> ", string, flags=re.I)
    string = re.sub(r"(?<!(?<![\dx])[\dA-Za-z])((\d)+(\,|\.)*(\d*)) *amps*", " <emeasure> ", string, flags=re.I)
    string = re.sub(r"(?<![\dA-Za-z])((\d)+(\,|\.)*(\d*)) *mah", " <emeasure> ", string, flags=re.I)
    string = re.sub(r"(?<![\dA-Za-z])((\d)+(\,|\.)*(\d*)) *vol.", " <emeasure> ", string, flags=re.I)
    string = re.sub(r"\b(?<![\dA-Za-z])((\d)+(\,|\.)*(\d*)) *kw\b", " <emeasure> ", string, flags=re.I)
    string = re.sub(r"\b(?<![\dA-Za-z])((\d)+(\,|\.)*(\d*)) *v+\b", " <emeasure> ", string, flags=re.I)
    string = re.sub(r"(?<![\dA-Za-z])((\d)+(\,|\.)*(\d*)) *volts*", " <emeasure> ", string, flags=re.I)
    string = re.sub(r"(?<![\dA-Za-z])((\d)+(\,|\.)*(\d*)) *w(ts)*(?![\dA-Za-z])", " <emeasure> ",
                    string, flags=re.I)

    # Pressure
    string = re.sub(r"(?<![\dA-Za-z])((\d)+(\,|\.)*(\d*)) *psi", " <pmeasure> ", string, flags=re.I)
    string = re.sub(r"(?<![\dA-Za-z])((\d)+(\,|\.)*(\d*)) *bar", " <pmeasure> ", string, flags=re.I)

    # Weights
    string = re.sub(r"(?<![\dA-Za-z])((\d)+(\,|\.)*(\d*)) *kgs*", " <wmeasure> ", string, flags=re.I)
    string = re.sub(r"(?<![\dA-Za-z])((\d)+(\,|\.)*(\d*)) *kilos*", " <wmeasure> ", string, flags=re.I)
    string = re.sub(r"\b(?<![\dA-Za-z])((\d)+(\,|\.)*(\d*)) *g\b", " <wmeasure> ", string, flags=re.I)

    # IT
    string = re.sub(r"(?<![\dA-Za-z])((\d)+(\,|\.)*(\d*)) *tb", " <itmeasure> ", string, flags=re.I)
    string = re.sub(r"(?<![\dA-Za-z])((\d)+(\,|\.)*(\d*)) *gb", " <itmeasure> ", string, flags=re.I)

    # Volume
    string = re.sub(r"(?<![\dA-Za-z])((\d)+(\,|\.)*(\d*)) *cc(?![0-9])", " <vmeasure> ", string, flags=re.I)
    string = re.sub(r"(?<![\dA-Za-z])((\d)+(\,|\.)*(\d*)) *litros*", " <vmeasure> ", string, flags=re.I)
    string = re.sub(r"(?<![\dA-Za-z])((\d)+(\,|\.)*(\d*))litrs*", " <vmeasure> ", string, flags=re.I)
    string = re.sub(r"\b(?<![\dA-Za-z])((\d)+(\,|\.)*(\d*)) *l+\b", " <vmeasure> ", string, flags=re.I)
    string = re.sub(r"(?<![\dA-Za-z])((\d)+(\,|\.)*(\d*)) *mls*", " <vmeasure> ", string, flags=re.I)
    string = re.sub(r"(?<![\dA-Za-z])((\d)+(\,|\.)*(\d*)) *ltrs*", " <vmeasure> ", string, flags=re.I)

    # Horse power
    string = re.sub(r"\b(?<![\dA-Za-z])(\d)+ *cv\b", " <hpmeasure> ", string, flags=re.I)
    string = re.sub(r"\b(?<![\dA-Za-z])((\d)+(\,|\.)*(\d*))+ *hp\b", " <hpmeasure> ", string, flags=re.I)

    # Time
    string = re.sub(r"\b(?<![\dA-Za-z])((\d)+(\:)*(\d*)) *hs*(?![\d])\b", " <time> ", string, flags=re.I)

    # Quantity
    string = re.sub(r"\bX\d{1,4}\b", " <quantity> ", string, flags=re.I)

    # Money
    string = re.sub(r"(?<![\dA-Za-z])\$ *((\d)+(\,|\.)*(\d*))", " <money> ", string, flags=re.I)

    # Dimension (could be smeasure too)
    string = re.sub(r"(((\d)+(\,|\.)*(\d*))+ {0,1}(x|\*){1} {0,1}((\d)+(\,|\.)*(\d*))+)"
                    "+( {0,1}(x|\*){1} {0,1}((\d)+(\,|\.)*(\d*))+)*", " <dimension> ", string, flags=re.I)

    # Resolution
    string = re.sub(r"\b(?<![A-Za-z\-])\d+p\b", " <res> ", string)

    # Date
    string = re.sub(r"\b\d{2}-\d{2}-(19\d{2}|20\d{2})\b", " <date> ", string)

    # Model
    string = re.sub(r"(?<!\d{4})[A-Za-z\-]+\d+[A-Za-z\-\.0-9]*", " <model> ", string, flags=re.I)
    string = re.sub(r"[A-Za-z\-\.0-9]*\d+[A-Za-z\-](?!\d{4})", " <model> ", string, flags=re.I)
    string = re.sub(r"<model> \d+", " <model> ", string, flags=re.I)
    string = re.sub(r"\d+ <model>", " <model> ", string, flags=re.I)

    # Years
    string = re.sub(r"(?<![A-Za-z0-9])19\d{2}|20\d{2}(?![A-Za-z0-9])", " <year> ", string)

    # Numbers
    string = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " <number> ", string)

    # String cleanup
    string = re.sub(r",", " , ", string)
    string = re.sub(r"/", " / ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\*", " * ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"#\S+", " <hashtag> ", string)
    string = re.sub(r"\\", " ", string)
    string = re.sub(r"\+", " ", string)
    string = re.sub(r"\d+", " <number> ", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`<>]", " ", string, flags=re.I)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()


def text_stemming(sentences, lang='english'):
    try:
        stemmer = SnowballStemmer(lang)
    except Exception:
        stemmer = SnowballStemmer('spanish')

    out_text = [[stemmer.stem(i) for i in text] for text in sentences]

    return out_text