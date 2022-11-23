from statistics import mean
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from string import punctuation
from collections import Counter
import pandas as pd


class Preprocessing:
    def __init__(self, code, flines):
        self.code = code # a piece of source_code
        self.flines = flines # the complete column of source_codes

    def create_metrics(self):
        """
        Creates metrics needed for modeling
        :return: returns an Array with metrics calculated from the source code
        """
        # TODO: assign vars for redundant code
        # TODO: self.var for all variable assignment

        # Number of characters in code/ length of code
        code_len = len(self.code)

        # Number of characters in line
        code_tokens = self.code.split("\n")
        nchar_in_line = mean([code_len for token in code_tokens])

        # Lines of code / file length in chars
        n_lines = self.code.count('\n')

        # Create n-grams
        #unigram = word2ngrams(text = self.code, n = 1)
        #bigram = word2ngrams(text = self.code, n = 2)
        #trigram = word2ngrams(text = self.code, n = 3)

        # Number of words in the text / file length in characters
        word_len_ratio = len(self.code.split())/code_len

        # Average number of words per line / file length in characters
        pre = (self.code.split('\n'))
        words_list = [x.split() for i, x in enumerate(pre)]
        words_count = [len(words_list[i]) for i, x in enumerate(pre)]
        avg_words_per_line = np.mean(words_count)

        # Number of whitespace / file length in characters
        whitespace_ratio = self.code.count(' ')/code_len

        # Number of line breaks / file length in characters
        linebreak_ratio = self.code.count('\n')/len(self.code)

        # Number of indentations / file length in characters
        indent_ratio = self.code.count('\t')/code_len

        # Number of upper_case words / file length in characters
        uppercase_ratio = sum(1 for char in self.code if char.isupper())/code_len

        # Number of lower_case words / file length in characters
        lowercase_ratio = sum(1 for char in self.code if char.islower())/code_len

        # Number of punctuation symbols / file length in characters
        punctuation_count = Counter(punc for line in self.code for punc in line if punc in punctuation)
        punctuation_count_ratio = sum(punctuation_count.values())/ code_len

        # Average length of words(exluding punctuation, excluding single letter or number)
        code_no_punc = code
        for punc in punctuation:
            code_no_punc = code_no_punc.replace(punc, '')
        code_no_single_char = ' '.join([w for w in code_no_punc.split() if len(w)>1])
        list_len_words = [len(x) for i, x in enumerate(code_no_single_char.split())]
        avg_char_per_word = np.mean(list_len_words)

        #Keywords count ('if', 'else', 'while', 'for', 'in', 'elif', 'or', 'not', 'with', 'and', 'is')
        keyword_count_ratio = sum([code.count(x) for x in ['if', 'else', 'while', 'for', 'in', 'elif', 'or', 'not', 'with', 'and', 'is'] ]) / code_len

        # Return all metrics as array
        return np.array([code_len,
                        nchar_in_line,
                        n_lines,
                        word_len_ratio,
                        avg_words_per_line,
                        whitespace_ratio,
                        linebreak_ratio,
                        indent_ratio,
                        uppercase_ratio,
                        lowercase_ratio,
                        punctuation_count_ratio,
                        avg_char_per_word,
                        keyword_count_ratio])


    # Word frequencies (unigram, bigram, trigram)
    #  We need to do this on the wole dataset.

    # TODO: implement n_gram range, and syncronize with analyzer param,
    # so we can build any kind of count_vectoriation with this function (character ngram, word unigram)
    def count_vectorizer(self, flines, analyzer = 'word', ngram_range = (1, 1), min_df = 3):
        """
        Apply count vectorizer to a given column (flines)
        :param analyzer:
            if 'word': vectorization happens with word n-grams
            if 'char': vectorization happens with character n-grams, padding the empty space to the tokens
            if 'char_wb': creates character n-grams only from text inside word boundaries
        :param ngram_range: the lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted.
            examples: (1, 1) extracts unigrams, (1, 2) extracts unigrams and bigrams
        :param min_df: remove terms with frequency lower than threshold
        :return: array of count_frequencies
        """
        count_vectorizer = CountVectorizer(analyzer = analyzer, ngram_range = ngram_range)
        X = count_vectorizer.fit_transform(flines).toarray()
        return X


    def word2ngrams(self, text, n=3, exact=True):
        """
        Convert text into character ngrams
        """
        return ["".join(j) for j in zip(*[text[i:] for i in range(n)])]


if __name__ == "__main__":
    # define code & flines
    code = """
            'import java.util.Scanner;\n \n public class EndlessKnight\n {\n \tpublic static void main(String[] args)\n \t{\n \t\tScanner in = new Scanner(System.in);\n \t\tint numCases = in.nextInt();\n \t\tfor (int i = 0; i < numCases; i++)\n \t\t\tdoCase(i + 1, in);\n \t}\n \n \tprivate static void doCase(int caseNum, Scanner in)\n \t{\n \t\tint H = in.nextInt();\n \t\tint W = in.nextInt();\n \t\tint R = in.nextInt();\n \t\t\n \t\tint[][] memoized = new int[W][H];\n \t\tfor (int[] arr : memoized)\n \t\t\tfor (int i = 0; i < arr.length; i++)\n \t\t\t\tarr[i] = -1;\n \t\t\n \t\tfor (int i = 0; i < R; i++)\n \t\t{\n \t\t\tint rockY = in.nextInt();\n \t\t\tint rockX = in.nextInt();\n \t\t\tmemoized[W - rockX][H - rockY] = 0;\n \t\t}\n \t\t\n \t\tint total = findMoves(W - 1, H - 1, memoized);\n \t\t\n \t\tSystem.out.println("Case #" + caseNum + ": " + total);\n \t}\n \n \tprivate static int findMoves(int i, int j, int[][] memoized)\n \t{\n \t\tif (i == 0 && j == 0)\n \t\t\treturn 1;\n \t\t\n \t\tif (i < 1 || j < 1)\n \t\t\treturn 0;\n \t\t\n \t\tif (memoized[i][j] != -1)\n \t\t\treturn memoized[i][j];\n \t\t\n \t\tmemoized[i][j] = (findMoves(i - 1, j - 2, memoized) + findMoves(i - 2, j - 1, memoized)) % 10007;\n \t\treturn memoized[i][j];\n \t}\n }\n'
           """

    data = pd.read_csv("/Users/timcerta/code/jbaccarin/xref/raw_data/gcj2008.csv")
    data = data[data.flines.notna()]
    data_small = data[:5]
    flines = data_small["flines"]

    prep = Preprocessing(code = code, flines = flines)
    res = prep.create_metrics()
    countvec = prep.count_vectorizer(flines)
    print(res)
