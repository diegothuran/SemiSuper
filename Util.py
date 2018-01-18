# -*- coding: utf-8 -*-
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from nltk import stem
from nltk.corpus import stopwords
from sklearn import preprocessing
import pandas as pd
import csv


def read_csv(path_to_csv, cols_to_use=[], data_col=0, label_cols=1):
    """
    Método lê um arquivo csv
    :param path_to_csv: endereço para o arquivo
    :param cols_to_use: colunas do arquivo que devem ser lidas
    :param data_col: coluna dos dados a serem classificados
    :param label_cols: coluna das labels
    :return: o dataset com os arquivos a serem classificados e um array com as labels
    """
    dataset = pd.read_csv(path_to_csv, usecols=cols_to_use)
    dataset = dataset.values
    return dataset[:, data_col], dataset[:, label_cols]


def read_csv_to_label(path_to_csv):
    """
    Lê um arquivo csv e retorna um array com os dados do arquivo
    :param path_to_csv: endereço para o arquivo
    :return: array com os dados do arquivo
    """
    dataset = pd.read_csv(path_to_csv)
    dataset = dataset.values

    return dataset.tolist()


def join_strings(list_of_strings):
    """
    Une uma lista de strings em uma única string
    :param list_of_strings: lista de strings
    :return: string com as strings unidas
    """
    return " ".join(list_of_strings)


def tokenize(text):
    """
        Aplica o processo de steeming em uma string e retira as pontuações
    :param text: texto para ser aplicado o processo e steeming
    :return: string depois do processo de stemming
    """
    steemming = stem.RSLPStemmer()
    tokens = text.decode('utf8').split()
    tokens = [i for i in tokens if i not in string.punctuation]
    stems = [steemming.stem(token) for token in tokens]
    stems = join_strings(stems)
    return stems


def vectorize_database_tfidf(database):
    """
        Extrai features de uma base de dados utilizando o processo de n_gram e tfidf
    :param database: base de dados
    :return: vetor de características e o vectorizer se for necessário aplicar em outra string
    """
    database = map(lambda x: x.lower(), database)
    database_ = map(lambda x: tokenize(x), database)

    pt_stop_words = set(stopwords.words('portuguese'))

    vectorizer = TfidfVectorizer(max_df=0.75, max_features=5000, lowercase=False, min_df=2,
                                 stop_words=pt_stop_words, ngram_range=(1, 4),
                                 use_idf=True)
    data = vectorizer.fit_transform(database_)

    return data.todense(), vectorizer


def encoding_labels(labels, labels_to_encode):
    """
    transforma um conjunto de labels do formato de strings para integer
    :param labels: conjunto de labels a serem transformadas
    :param labels_to_encode: vetor com um exemplo de cada label presente no conjunto
    :return: array com o conjunto de labels em formato de inteiros
    """
    le = preprocessing.LabelEncoder()
    le.fit(labels_to_encode)
    return le.transform(labels)


def write_csv(features_vector, labels, file_name):
    """
    Escreve um csv com os dados novos gerados
    :param features_vector: vetor com os dados a serem gravados
    :param labels: array com os labels
    :param file_name: nome do arquivo a ser escrito
    """
    with open(file_name, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for row, label in zip(features_vector, labels):
            spamwriter.writerow(row + label)

def generate_new_data(database, labels, data_w_no_label, classifier):

    """
        Gera, a partir de uma base de dados sem rótulos, os rótulos dessa base
    :param database: base de dados sem rotulada
    :param labels: rótulos da base de dados previamente rotulada
    :param data_w_no_label: base de dados sem rótulos
    :param classifier: classificador a ser usado
    :return:
    """
    feature_data, vectorizer = vectorize_database_tfidf(database)

    labels = labels.tolist()

    del (labels[0])
    del (labels[0])
    labels = np.array(labels)
    classifier = classifier.fit(feature_data, labels)

    new_data, _ = vectorize_database_tfidf(data_w_no_label)

    classifications = classifier.predict(new_data)

    write_csv(data_w_no_label, classifications, "generated_labels.csv")



if __name__ == "__main__":

    database, labels = read_csv('database.csv', cols_to_use=[0, 1], data_col=0, label_cols=1)

    database = sorted(database)
    del (database[0])
    del (database[0])
    labels_to_encode = ['Product', '0', 'Pro', 'Prod', ' ']

    labels = encoding_labels(labels, labels_to_encode)

    data_w = read_csv_to_label('products.csv')

    clf = LinearSVC()

    generate_new_data(database, labels, data_w, clf)