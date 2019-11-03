########################################
## CS447 Natural Language Processing  ##
##           Homework 2               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Train a bigram HMM for POS tagging
##
from collections import defaultdict
from math import log, exp
import numpy as np
import os.path
import sys

# Unknown word token
UNK = 'UNK'

class TaggedWord:
    """
    Class that stores a word and tag together
    """
    def __init__(self, taggedString, lowercase=False):
        parts = taggedString.split('_')
        self.word = parts[0] # convert all words to lower case
        self.tag = parts[1]

        if lowercase:
            self.word = self.word.lower()


class HMM:
    """
    Class definition for a bigram HMM
    """
    def __init__(self, unknownWordThreshold=5):
        self.minFreq = unknownWordThreshold # words occuring fewer than `unknownWordThreshold` times should be treated as UNK
        
        # compute following from train data
        self.states = set()
        self.n_tags = 0
        self.vocab = set() 
        self.tag_counts = defaultdict(float) 
        self.bigram_tag_counts = defaultdict(float) 
        self.word_tag_counts = defaultdict(float) 
        self.init_log_tag_prob = defaultdict(float)

    def readLabeledData(self, inputFile, lowercase=True):
        """
        Reads a labeled data inputFile, and returns a nested list of sentences, where each sentence is a list of TaggedWord objects
        """
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = []
            for line in file:
                raw = line.split()
                sen = []
                for token in raw:
                    sen.append(TaggedWord(token, lowercase))
                sens.append(sen) # append this list as an element to the list of sentences
            return sens
        else:
            print("Error: unlabeled data file %s does not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script

    def readUnlabeledData(self, inputFile):
        """
        Reads an unlabeled data inputFile, and returns a nested list of sentences, where each sentence is a list of strings
        """
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = []
            for line in file:
                sen = line.split() # split the line into a list of words
                sens.append(sen) # append this list as an element to the list of sentences
            return sens
        else:
            print("Error: unlabeled data file %s does not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script

    def compute_vocab_and_states(self, train_data):
        word_freq = defaultdict(int)
        self.vocab.add(UNK)

        for sen in train_data:
            for tagged_word in sen:
                word_freq[tagged_word.word] = word_freq[tagged_word.word] + 1
        
        for sen in train_data:
            for tagged_word in sen:
                self.states.add(tagged_word.tag)
                if word_freq[tagged_word.word] > self.minFreq:
                    self.vocab.add(tagged_word.word)
        
        self.n_tags = len(self.states)

    def train(self, trainFile):
        """
        Given labeled corpus in trainFile, build the HMM distributions from the observed counts
        """
        train_data = self.readLabeledData(trainFile) # train_data is a nested list of TaggedWords
        self.compute_vocab_and_states(train_data)
        
        for sen in train_data:
            for tagged_word in sen:
                if not tagged_word.word in self.vocab:
                    tagged_word.word = UNK
            
            self.tag_counts[sen[0].tag] += 1.0
            self.word_tag_counts[(sen[0].word, sen[0].tag)] += 1.0
            for i in range(1, len(sen)):
                self.tag_counts[sen[i].tag] += 1.0
                self.word_tag_counts[(sen[i].word, sen[i].tag)] += 1.0
                self.bigram_tag_counts[(sen[i-1].tag, sen[i].tag)] += 1.0
        
        normalizer = 0
        for tag in self.states:
            self.init_log_tag_prob[tag] = log(self.tag_counts[tag])
            normalizer += self.tag_counts[tag]
        normalizer = log(normalizer)
        for tag in self.states:
            self.init_log_tag_prob[tag] -= normalizer

    def get_log_transition_prob(self, tag, prev_tag):
        """
        return log[smoothed p(tag | prev_tag)]
        """
        return log ( (self.bigram_tag_counts[(prev_tag, tag)] + 1) / (self.tag_counts[tag] + self.n_tags) )

    def get_log_emission_prob(self, word, tag):
        """
        return log[p(word | tag)]
        """
        if self.word_tag_counts[(word, tag)] == 0.0:
            return float('-inf')
        return log( (self.word_tag_counts[(word, tag)]) / (self.tag_counts[tag]) )

    def test(self, testFile, outFile):
        """
        Given an unlabeled corpus in testFile, output the Viterbi tag sequences as a labeled corpus in outFile
        """
        data = self.readUnlabeledData(testFile)

        f=open(outFile, 'w+')
        for sen in data:
            processed_sen = []
            for word in sen:
                if word.lower() not in self.vocab:
                    processed_sen.append(UNK)
                else:
                    processed_sen.append(word.lower())

            vitTags = self.viterbi(processed_sen)
            senString = ''
            for i in range(len(sen)):
                senString += sen[i]+"_"+vitTags[i]+" "
            print(senString.rstrip(), end="\n", file=f)

    def viterbi(self, words):
        """
        Given a list of words, runs the Viterbi algorithm and returns a list containing the sequence of tags
        that generates the word sequence with highest probability, according to this HMM.
        Returns: the list of Viterbi POS tags (strings)
        """
        tags = sorted(list(self.states))
        n_words = len(words)
        trellis = np.zeros((n_words, self.n_tags), dtype=np.float64)
        trellis.fill(np.inf)
        backpointers = np.zeros((n_words, self.n_tags), dtype=np.int32)

        for j in range(self.n_tags):
            trellis[0, j] = self.init_log_tag_prob[tags[j]] + self.get_log_emission_prob(words[0], tags[j])
        
        for i in range(1, n_words):
            for j in range(self.n_tags):
                for t in range(self.n_tags):
                    temp = trellis[i-1, t] + self.get_log_transition_prob(tags[j], tags[t])
                    if temp > trellis[i, j] or trellis[i, j] == np.inf:
                        trellis[i, j] = temp
                        backpointers[i, j] = t
                trellis[i, j] += self.get_log_emission_prob(words[i], tags[j])
        
        best_end_tag = -1
        best_log_prob = float('-inf')
        for j in range(self.n_tags):
            if trellis[-1, j] > best_log_prob:
                best_end_tag = j
                best_log_prob = trellis[-1, j]
        
        tag_seq = []
        i = n_words - 1
        while i >= 0:
            tag_seq.append(tags[best_end_tag])
            best_end_tag = backpointers[i, best_end_tag]
            i -= 1

        tag_seq.reverse()
        return tag_seq


if __name__ == "__main__":
    print('Training model...')
    tagger = HMM()
    tagger.train('train.txt')

    print('Testing...')
    tagger.test('test.txt', 'out.txt')

    print('Done!')
