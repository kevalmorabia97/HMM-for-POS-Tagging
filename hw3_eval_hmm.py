########################################
## CS447 Natural Language Processing  ##
##           Homework 2               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Evaluate the output of your bigram HMM POS tagger
##
import numpy as np
import os
import sys

class TaggedWord:
    """
    Class that stores a word and tag together
    """
    def __init__(self, taggedString):
        parts = taggedString.split('_')
        self.word = parts[0] # convert all words to lower case
        self.tag = parts[1]


class Eval:
    """
    A class for evaluating POS-tagged data
    """
    def __init__(self, goldFile, testFile):
        gold_sens = self.readLabeledData(goldFile)
        test_sens = self.readLabeledData(testFile)
        
        self.tags = set()
        for sen in gold_sens:
            for tagged_word in sen:
                self.tags.add(tagged_word.tag)
        
        self.tags = sorted(self.tags)
        n_tags = len(self.tags)
        self.tag_to_ind = {t:i for i,t in enumerate(self.tags)}
        self.ind_to_tag = {i:t for t,i in self.tag_to_ind.items()}

        total_sens = len(gold_sens)
        correct_sens = 0.0
        self.confusion_matrix = np.zeros((n_tags, n_tags), dtype=np.int32)
        for i in range(len(gold_sens)):
            for j in range(len(gold_sens[i])):
                self.confusion_matrix[self.tag_to_ind[gold_sens[i][j].tag], self.tag_to_ind[test_sens[i][j].tag]] += 1
            
            if [tw.tag for tw in gold_sens[i]] ==  [tw.tag for tw in test_sens[i]]:
                correct_sens += 1
        self.sentence_acc = correct_sens / total_sens

    def getTokenAccuracy(self):
        """
        Return the percentage of correctly-labeled tokens
        """
        return self.confusion_matrix.diagonal().sum()/self.confusion_matrix.sum()

    def getSentenceAccuracy(self):
        """
        Return the percentage of sentences where every word is correctly labeled
        """
        return self.sentence_acc

    def writeConfusionMatrix(self, outFile):
        """
        Write a confusion matrix to outFile
        """
        with open(outFile, 'w') as f:
            f.write(','.join(self.tags) + '\n')
            for t in self.tags:
                f.write(t + ',' + ','.join(self.confusion_matrix[self.tag_to_ind[t]].astype(str)) + '\n')

    def getPrecision(self, tagTi):
        """
        Return the tagger's precision when predicting tag t_i
        """
        ind = self.tag_to_ind[tagTi]
        return self.confusion_matrix[ind, ind] / self.confusion_matrix[:, ind].sum()

    def getRecall(self, tagTj):
        """
        Return the tagger's recall for correctly predicting gold tag t_j
        """
        ind = self.tag_to_ind[tagTj]
        return self.confusion_matrix[ind, ind] / self.confusion_matrix[ind].sum()

    def readLabeledData(self, inputFile):
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
                    sen.append(TaggedWord(token))
                sens.append(sen) # append this list as an element to the list of sentences
            return sens
        else:
            print("Error: unlabeled data file %s does not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script


if __name__ == "__main__":
    # Pass in the gold and test POS-tagged data as arguments
    if len(sys.argv) < 2:
        print("Call hw2_eval_hmm.py with two arguments: gold.txt and out.txt")
    else:
        gold = sys.argv[1]
        test = sys.argv[2]
        # You need to implement the evaluation class
        eval = Eval(gold, test)
        # Calculate accuracy (sentence and token level)
        print("Token accuracy: ", eval.getTokenAccuracy())
        print("Sentence accuracy: ", eval.getSentenceAccuracy())
        # Calculate recall and precision
        print("Recall on tag NNP: ", eval.getPrecision('NNP'))
        print("Precision for tag NNP: ", eval.getRecall('NNP'))
        # Write a confusion matrix
        eval.writeConfusionMatrix("confusion_matrix.txt")
