import csv
import os
import nltk
from nltk.collocations import *
from parse_debates import DebateParser
from nltk.tokenize import RegexpTokenizer

def removeNonAscii(s):
    return "".join(filter(lambda x: ord(x)<128, s))


def get_bigram_likelihood(statements, freq_filter=3, nbest=200):
    """
    Returns n (likelihood ratio) bi-grams from a group of documents
    :param        statements: list of strings
    :param output_file: output path for saved file
    :param freq_filter: filter for # of appearances in bi-gram
    :param       nbest: likelihood ratio for bi-grams
    """

    words = list()
    print 'Generating word list...'
    for statement in statements:
        # remove non-words
        tokenizer = RegexpTokenizer(r'\w+')
        words.extend(tokenizer.tokenize(statement))

    bigram_measures = nltk.collocations.BigramAssocMeasures()
    bigram_finder = BigramCollocationFinder.from_words(words)

    # only bi-grams that appear n+ times
    bigram_finder.apply_freq_filter(freq_filter)

    # TODO: use custom stop words
    bigram_finder.apply_word_filter(lambda w: len(w) < 3 or w.lower() in nltk.corpus.stopwords.words('english'))

    bigram_results = bigram_finder.nbest(bigram_measures.likelihood_ratio, nbest)

    return bigram_finder.score_ngrams(bigram_measures.likelihood_ratio)


def save_bigram_likelihood_tsv(statements, path):
    """
    saves likely bigrams in a tsv
    :param        statements: list of strings
    :param path: output path for saved tsv
    """
    with open(path, "wb+") as output_file:
        writer = csv.writer(output_file, delimiter="\t")

        if len(statements) > 0:
            statements = [removeNonAscii(statement) for statement in statements]
            ngrams = get_bigram_likelihood(statements)
            if ngrams != '':
                for ngram in ngrams:
                    writer.writerow([ngram[0][0] + '_' + ngram[0][1], ngram[1]])

def save_bigrams_for_replacement_file_txt(statements, path):
    """
    saves likely bigrams in a txt
    :param        statements: list of strings
    :param path: output path for saved txt
    """
    with open(path, "wb+") as output_file:
        if len(statements) > 0:
            statements = [removeNonAscii(statement) for statement in statements]
            ngrams = get_bigram_likelihood(statements)
            if ngrams != '':
                for ngram in ngrams:
                    output_file.write(ngram[0][0] + ' ' + ngram[0][1] + '\n')

if __name__ == "__main__":
    parser = DebateParser("./data/debates")
    parser.parse()
    save_bigram_likelihood_tsv([item[0] for sublist in parser.statements.values() for item in sublist],
                               os.path.join("data", "ngrams.tsv"))
    save_bigrams_for_replacement_file_txt([item[0] for sublist in parser.statements.values() for item in sublist],
                                      os.path.join("./data/mallet_files", "replacements.txt"))