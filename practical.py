from Corpora import MovieReviewCorpus
from Lexicon import SentimentLexicon
from Statistics import SignTest
from Classifiers import NaiveBayesText, SVMText
from Extensions import SVMDoc2Vec

# retrieve corpus
corpus=MovieReviewCorpus(stemming=False,pos=False)

# use sign test for all significance testing
signTest=SignTest()

# location of svmlight binaries 
# TODO: change this to your local installation
svmlight_dir="/path/to/svmlight/binaries/"

print "--- classifying reviews using sentiment lexicon  ---"

# read in lexicon
lexicon=SentimentLexicon()

# on average there are more positive than negative words per review (~7.13 more positive than negative per review)
# to take this bias into account will use threshold (roughly the bias itself) to make it harder to classify as positive
threshold=8

# question 0.1
lexicon.classify(corpus.reviews,threshold,magnitude=False)
token_preds=lexicon.predictions
print "token-only results: %.2f" % lexicon.getAccuracy()

lexicon.classify(corpus.reviews,threshold,magnitude=True)
magnitude_preds=lexicon.predictions
print "magnitude results: %.2f" % lexicon.getAccuracy()

# question 0.2
p_value=signTest.getSignificance(token_preds,magnitude_preds)
print "magnitude lexicon results are",("significant" if p_value < 0.05 else "not significant"),"with respect to token-only","(p=%.8f)" % p_value

# question 1.0
print "--- classifying reviews using Naive Bayes on held-out test set ---"
NB=NaiveBayesText(smoothing=False,bigrams=False,trigrams=False,discard_closed_class=False)
NB.train(corpus.train)
NB.test(corpus.test)
# store predictions from classifier
non_smoothed_preds=NB.predictions
print "Accuracy without smoothing: %.2f" % NB.getAccuracy()

# question 2.0
# use smoothing
NB=NaiveBayesText(smoothing=True,bigrams=False,trigrams=False,discard_closed_class=False)
NB.train(corpus.train)
NB.test(corpus.test)
smoothed_preds=NB.predictions
# saving this for use later
num_non_stemmed_features=len(NB.vocabulary)
print "Accuracy using smoothing: %.2f" % NB.getAccuracy()

# question 2.1
# see if smoothing significantly improves results
p_value=signTest.getSignificance(non_smoothed_preds,smoothed_preds)
print "results using smoothing are",("significant" if p_value < 0.05 else "not significant"),"with respect to no smoothing","(p=%.8f)" % p_value

# question 3.0
print "--- classifying reviews using 10-fold cross-evaluation ---"
# using previous instantiated object
NB.crossValidate(corpus)
# using cross-eval for smoothed predictions from now on
smoothed_preds=NB.predictions
print "Accuracy: %.2f" % NB.getAccuracy()
print "Std. Dev: %.2f" % NB.getStdDeviation()

# question 4.0
print "--- stemming corpus ---"
# retrieve corpus with tokenized text and stemming (using porter)
stemmed_corpus=MovieReviewCorpus(stemming=True,pos=False)
print "--- cross-validating NB using stemming ---"
NB.crossValidate(stemmed_corpus)
stemmed_preds=NB.predictions
print "Accuracy: %.2f" % NB.getAccuracy()
print "Std. Dev: %.2f" % NB.getStdDeviation()

# TODO Q4.1
# see if stemming significantly improves results on smoothed NB

# TODO Q4.2
print "--- determining the number of features before/after stemming ---"

# question Q5.0
# cross-validate model using smoothing and bigrams
print "--- cross-validating naive bayes using smoothing and bigrams ---"
NB=NaiveBayesText(smoothing=True,bigrams=True,trigrams=False,discard_closed_class=False)
NB.crossValidate(corpus)
smoothed_and_bigram_preds=NB.predictions
print "Accuracy: %.2f" % NB.getAccuracy()
print "Std. Dev: %.2f" % NB.getStdDeviation()

# see if bigrams significantly improves results on smoothed NB only
p_value=signTest.getSignificance(smoothed_preds,smoothed_and_bigram_preds)
print "results using smoothing and bigrams are",("significant" if p_value < 0.05 else "not significant"),"with respect to smoothing only","(p=%.8f)" % p_value

# TODO Q5.1

# TODO Q6 and 6.1
print "--- classifying reviews using SVM 10-fold cross-eval ---"

# TODO Q7
print "--- adding in POS information to corpus ---"
print "--- training svm on word+pos features ----"
print "--- training svm discarding closed-class words ---"

# question 8.0
print "--- using document embeddings ---"
