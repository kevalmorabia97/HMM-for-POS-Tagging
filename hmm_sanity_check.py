########################################
## CS447 Natural Language Processing  ##
##           Homework 2               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Validation script for HMM POS tagging code (Part 1)
##
import sys, traceback

if __name__ == "__main__":
    # HMM test script
    try:
        print("---")
        print("Sanity-checking HMM implementation...")
        # Import your solution
        import hw3_hmm as hmm
        # Get your HMM
        myHMM = hmm.HMM(4) # set UNK threshold to 4, rather than default 5
        # Train your HMM on labeled data
        myHMM.train('train.txt')
        # Evaluate your tagger on a sample sentence
        sampleSen = "This is a sample sentence".split()
        tags = myHMM.viterbi(sampleSen)
        tagged = ''
        for i in range(len(sampleSen)):
            tagged += sampleSen[i]+"_"+tags[i]+" "
        goldTags = ["DT", "VBZ", "DT", "JJ", "NN"]
        if goldTags == tags:
            print("INFO: Your HMM tagged the sample sentence correctly, so there's a non-zero chance that your implementation is correct.")
        else:
            print("ERROR: Your HMM did not tag the sample sentence correctly.")

        print("---")
        print("Sanity-checking HMM evaluation implementation...")

        # HMM-eval test script
        import hw3_eval_hmm as ev
        goldFile = "gold.txt"
        outFile = "out.txt" # test script runs on out.txt, the predicted tags from your HMM
        # Get your Eval object, assuming both files exist
        eval = ev.Eval(goldFile, outFile)
        tokenAcc = eval.getTokenAccuracy()
        if tokenAcc < 0.90:
            print("ERROR: Token accuracy for bigram HMM tagger seems a little low...")
        elif tokenAcc > 0.95:
            print("ERROR: Token accuracy for bigram HMM tagger seems a little high..")
        else:
            print("INFO: Token accuracy for bigram HMM is plausible (but, as always, verify your implementation)")
        senAcc = eval.getSentenceAccuracy() # No hints on sentence accuracy :(
        # Calculate recall and precision
        nnpPrec = eval.getPrecision('NNP')
        if nnpPrec < 0.80:
            print("ERROR: Precision for tag NNP seems a little low...")
        elif nnpPrec > 0.90:
            print("ERROR: Precision for tag NNP seems a little high..")
        else:
            print("INFO: Precision for tag NNP is plausible (but, as always, verify your implementation)")
        nnpRec = eval.getRecall('NNP')
        if nnpRec < 0.88:
            print("ERROR: Recall for tag NNP seems a little low...")
        elif nnpRec > 0.95:
            print("ERROR: Recall for tag NNP seems a little high..")
        else:
            print("INFO: Recall for tag NNP is plausible (but, as always, verify your implementation)")
        # Write a confusion matrix
        eval.writeConfusionMatrix("sanity_check_conf_matrix.txt")
    except Exception as e:
        print("ERROR: An exception was thrown while running your HMM code:")
        traceback.print_exc()
    else:
        # If you reach this clause, we can at least run your code
        print("SUCCESS: Your HMM code compiles and runs without throwing an unhandled exception (you may still need to verify correctness).")
