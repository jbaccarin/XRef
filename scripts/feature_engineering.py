from statistics import mean
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from string import punctuation
from collections import Counter


def create_metrics(code):

    # Number of characters in text
    nchar_in_code = len(code)

    # Number of characters in line
    code_tokens = code.split("\n")
    nchar_in_line = mean([len(token) for token in code_tokens])

    # Lines of code / file length in chars
    n_lines = code.count('\n')

    # Create n-grams
    unigram = word2ngrams(text = code, n = 1)
    bigram = word2ngrams(text = code, n = 2)
    trigram = word2ngrams(text = code, n = 3)

    # Number of words in the text / file length in characters
    word_len_ratio = len(code.split())/len(code)

    # Average number of words per line / file length in characters
    pre = (code.split('\n'))
    words_list = [x.split() for i, x in enumerate(pre)]
    words_count = [len(words_list[i]) for i, x in enumerate(pre)]
    average_words_per_line = np.mean(words_count)
    average_words_per_line

    # Number of whitespace / file length in characters
    whitespace_ratio = code.count(' ')/len(code)

    # Number of line breaks / file length in characters
    linebreak_ratio = code.count('\n')/len(code)

    # Number of indentations / file length in characters
    indent_ratio = code.count('\t')/len(code)

    # Number of upper_case words
    uppercase_ratio = sum(1 for char in code if char.isupper())/len(code)

    # Number of lower_case words
    lowercase_ratio = sum(1 for char in code if char.islower())/len(code)

    # Number of punctuation symbols
    punctuation_count = Counter(punc for line in code for punc in line if punc in punctuation)
    punctuation_count = sum(punctuation_count.values())/len(code)

    return nchar_in_code, nchar_in_line, n_lines, unigram, bigram, trigram, word_len_ratio, average_words_per_line, whitespace_ratio, linebreak_ratio, indent_ratio, uppercase_ratio, lowercase_ratio, punctuation_count


# Word frequencies
def count_vectorizer(flines, analyzer = None):
#    different, we need to do this on the wole dataset.
    count_vectorizer = CountVectorizer(analyzer = analyzer)
    X = count_vectorizer.fit_transform(flines).toarray()
    return X


def word2ngrams(text, n=3, exact=True):
    """ Convert text into character ngrams. """
    return ["".join(j) for j in zip(*[text[i:] for i in range(n)])]
