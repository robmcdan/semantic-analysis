import ast
import os
import subprocess
from multiprocessing import pool, cpu_count
import numpy
import csv
import numpy as np
import csv
import shutil
from scipy.sparse import coo_matrix
from pattern.text.en import modality, sentiment
import tempfile

# ensure these directories match mallet
mallet_bin = "./bin/mallet"
mallet_dir = "/Users/robm/Documents/Java/MalletGit/Mallet"
replacements = "/Users/robm/Documents/Lingistic/HowBiased/Resources/debates/GOP/mallet_files/replacements.txt"
sequences = "/Users/robm/Documents/Lingistic/HowBiased/Resources/debates/GOP/mallet_files/gop_text.sequences"


class MalletLDA:
    def __init__(self, doc_topics, doc_word_topic_counts, has_header=False):
        """
        Parameters
        ----------
        :doc_topics: string
            Path to document containing the topic composition of each training file
            generated in Mallet by using "--output-doc-topics".
        doc_word_topic_counts : string
            path to document containing sparse representation of topic-word assignments
            generated with "--word-topic-counts-file".
        doc_topic_keys: string
            path to document containing top words for each topic and any Dirichlet parameters
            generated with "--output-topic-keys""
        """
        self.avoid_header = has_header
        self.theta = load_theta(doc_topics, avoid_header=False)

        self._load_phi_and_vocabulary(doc_word_topic_counts)
        self.Z = self.phi.shape[0]
        self.num_topics = self.phi.shape[0]  # Number of topics.
        self.num_terms = self.phi.shape[1]  # Number of terms.


    def _item_description(self, i, **kwargs):
        """
        Yields proportion of each topic in document i.
        """
        return [(t, self.theta[i, t]) for t in xrange(self.theta.shape[1])]

    def _dimension_description(self, k, **kwargs):
        """
        Yields probability distribution over terms for document i.
        """
        return [(w, self.phi[k, w]) for w in xrange(self.phi.shape[1])]

    def _dimension_items(self, k, threshold, **kwargs):
        """
        Returns items that contain ``k`` at or above ``threshold``.

        Parameters
        ----------
        k : int
            Topic index.
        threshold : float
            Minimum representation of ``k`` in document.

        Returns
        -------
        description : list
            A list of ( item, weight ) tuples.
        """

        description = [(self.metadata[i]['id'], self.theta[i, k])
                       for i in xrange(self.theta[:, k].size)
                       if self.theta[i, k] >= threshold]
        return description


    def list_topic(self, k, Nwords=10):
        """
        Yields a list of the top ``Nwords`` for topic ``k``.
        """
        words = self._dimension_description(k)
        as_list = [(self.vocabulary[w], p) for w, p in words if p > 0.0]
        as_list.sort(key=lambda tup: tup[1], reverse=True)
        return [w for w, p in as_list[:Nwords]]


    def print_topic(self, k, Nwords=10):
        """
        Yields the top ``Nwords`` for topic ``k`` as a string.
        """
        as_string = ', '.join(self.list_topic(k, Nwords))
        print as_string


    def list_topics(self, Nwords=10):
        """
        Yields lists of the top ``Nwords`` for each topic.
        """
        as_dict = {}
        for k in xrange(self.Z):
            as_dict[k] = self.list_topic(k, Nwords)
        return as_dict


    def print_topics(self, Nwords=10):
        """
        Yields the top ``Nwords`` for each topic, as a string.
        """
        as_dict = self.list_topics(Nwords)
        s = []
        for key, value in as_dict.iteritems():
            s.append('{0}: {1}'.format(key, ', '.join(value)))
        as_string = '\n'.join(s)
        print as_string

    def _load_phi_and_vocabulary(self, word_topic_counts):
        """
        Reconstruct phi posterior distributions -- topic (rows) distributions over words (cols),
        and vocabulary index (map matrix indices for words to word-strings)
        """
        print 'loading phi'
        vocabulary = {}

        W = []
        T = []
        C = []
        es = None

        with open(word_topic_counts, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            for line in reader:
                w = int(line[0])
                term = str(line[1])
                vocabulary[w] = term
                for l in line[2:]:
                    k, c = l.split(':')
                    W.append(w)
                    T.append(int(k))
                    C.append(float(c))

        K = max(T) + 1
        V = len(set(W))

        phi = coo_matrix((C, (T, W)), shape=(K, V)).todense()

        # Normalize
        for k in xrange(K):
            phi[k, :] /= np.sum(phi[k, :])

        self.phi = phi
        self.vocabulary = vocabulary

def load_theta(doc_topics, avoid_header=True, chunk=False):
    """
    Reconstruct theta posterior distribution matrix (the proportion of each topic (col) in each document (row))
    """

    def chunks(source_list, n):
        for i in xrange(0, len(source_list), n):
            yield source_list[i:i + n]

    documents = []
    topics = []
    proportion = []

    with open(doc_topics, 'rb') as doc_topics_open:
        reader = csv.reader(doc_topics_open, delimiter='\t')

        for line in reader:
            if avoid_header is True:
                avoid_header = False
                continue  # Avoid header row, if present.

            document_id = int(line[0])
            print 'processing document {0}'.format(document_id)
            topic_distribution = line[2:]

            # some versions of mallet provide a chunked (topic,proportion) format for the doc_topics file
            if chunk:
                topic_distribution = list(chunks(topic_distribution, 2))

            distribution_length = len(topic_distribution)

            nonzero_topics = []

            for i in xrange(0, distribution_length):
                if topic_distribution[i] != '':  # na
                    if chunk:
                        nonzero_topics.append((topic_distribution[i][0], float(topic_distribution[i][1])))
                    else:
                        nonzero_topics.append((i, float(topic_distribution[i])))

            for topic_id, prop in nonzero_topics:
                documents.append(document_id)
                topics.append(topic_id)
                proportion.append(prop)

        num_documents = len(set(documents))  # Number of documents.
        num_topics = len(set(topics))  # Number of topics.

    return coo_matrix((proportion, (documents, topics)), shape=(num_documents, num_topics)).todense()

def get_topics(doc_topics, has_headers=False):
    """

    :param doc_topics: doc_topics file
    :param has_headers:
    :return:
    """
    theta = load_theta(doc_topics, has_headers, False)
    topics = [(i, theta[0:, i]) for i in xrange(theta.shape[1])]
    topics.sort(key=lambda tup: tup[1], reverse=True)
    return [(topic[0], topic[1].item(0)) for topic in topics]

def getTopicsForStatement(statement, num_topics=40):

    # setup some temp files for mallet to play with
    # the use of temp files allows thread safety on the sequence and inferencer files
    original_dir = os.getcwd()
    text_file_for_statment = tempfile.NamedTemporaryFile("wb+", delete=False)
    sequences_file = tempfile.NamedTemporaryFile("wb+", delete=False)
    doc_topics_file = tempfile.NamedTemporaryFile("wb+", delete=False)
    corpus_sequences_file = tempfile.NamedTemporaryFile("wb+", delete=False)
    inferencer_file = tempfile.NamedTemporaryFile("wb+", delete=False)

    try:
        text_file_for_statment.write(statement)
        text_file_for_statment.flush()
        text_file_for_statment.close()
        inferencer_file.close()
        shutil.copy("/Users/robm/Documents/Lingistic/HowBiased/Resources/debates/GOP/mallet_files/gop.inferencer", inferencer_file.name)
        corpus_sequences_file.close()
        shutil.copy(sequences, corpus_sequences_file.name)
        sequences_file.close()
        doc_topics_file.close()

        os.chdir(mallet_dir)

        # step 1: vectorize statement
        result = subprocess.call([mallet_bin,
                                  'import-file',
                                  '--input',
                                  '{0}'.format(text_file_for_statment.name),
                                  '--output',
                                  sequences_file.name,
                                  '--keep-sequence',
                                  '--replacement-files',
                                  replacements,
                                  '--remove-stopwords',
                                  '--use-pipe-from',
                                  corpus_sequences_file.name
                                  ])
        if result != 0:
            raise Exception('task.vectorize_JD_task: error code returned from Mallet: {0}'.format(result))

        # step 2: infer topics
        result = subprocess.call([mallet_bin,
                                      'infer-topics',
                                      '--inferencer',
                                      inferencer_file.name,
                                      '--input',
                                      sequences_file.name,
                                      '--output-doc-topics',
                                      doc_topics_file.name,
                                      '--num-iterations',
                                      '100'
                                      ])
        if result != 0:
                raise Exception('error code returned from Mallet: {0}'.format(result))

        topics = get_topics(doc_topics_file.name, True)
        os.chdir(original_dir)
    finally:
        os.chdir(original_dir)
        os.remove(sequences_file.name)
        os.remove(text_file_for_statment.name)
        os.remove(doc_topics_file.name)
        os.remove(corpus_sequences_file.name)
        os.remove(inferencer_file.name)

    return topics[:num_topics]

if __name__ == "__main__":
    model = MalletLDA('./Data/mallet_files/doc_topics.tsv', './Data/mallet_files/topic_counts.tsv', './Data/mallet_files/topic_keys.tsv')
