import nltk
import sys
from pprint import pformat

def tokenize(file):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    rawText = open(file).readlines()
    segmentedCorpus = []
    for line in rawText:
        sentence = nltk.word_tokenize(line)
        segmentedCorpus.append(sentence)
    return segmentedCorpus

if __name__ == "__main__":
    # Downloads the 'punkt' package from nltk_data, if you haven't downloaded
    # it already
    dl = nltk.downloader.Downloader("http://nltk.github.com/nltk_data/")
    dl.download('punkt')

    inputFile = "sentences.txt"
    outputFile = "hw3_cfg_out.txt"
    print("Parsing", inputFile, " to ", outputFile)


    out = open(outputFile, "w+")
    corpus = tokenize(inputFile)
    grammar = nltk.data.load('file:mygrammar.cfg')
    parser = nltk.ChartParser(grammar)
    i = 0
    for s in corpus:
        i += 1
        sentence = " ".join(s)
        parses = []
        parses = list(parser.parse_all(s))
        message = ""
        if len(parses) > 0:
            message = "SUCCESS (" + str(len(parses)) + " parses) "
            message = message.ljust(30)
        else:
            message = "FAILURE ".ljust(30)
        print(message + sentence)
        header = "########################################\n### Sentence " + \
                 str(i) + ": " + str(len(parses)) + " parses\n"
        out.write(header)
        sline = "### <" + sentence + ">\n"
        out.write(sline)
        out.write("########################################\n")

        if len(parses) > 0:
            j = 0
            for t in parses:
                j += 1
                p = "Parse " + str(j) + "\n"
                out.write(p)
                out.write(pformat(t))
                out.write("\n-------------------------\n")
