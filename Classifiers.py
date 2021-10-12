import os
from subprocess import call
from nltk.util import ngrams
from Analysis import Evaluation
import numpy as np
from sklearn import svm

class NaiveBayesText(Evaluation):
    def __init__(self,smoothing,bigrams,trigrams,discard_closed_class):
        """
        initialisation of NaiveBayesText classifier.

        @param smoothing: use smoothing?
        @type smoothing: booleanp

        @param bigrams: add bigrams?
        @type bigrams: boolean

        @param trigrams: add trigrams?
        @type trigrams: boolean

        @param discard_closed_class: restrict unigrams to nouns, adjectives, adverbs and verbs?
        @type discard_closed_class: boolean
        """
        self.reset()
        # use smoothing?
        self.smoothing=smoothing
        # add bigrams?
        self.bigrams=bigrams
        # add trigrams?
        self.trigrams=trigrams
        # restrict unigrams to nouns, adjectives, adverbs and verbs?
        self.discard_closed_class=discard_closed_class
        # stored predictions from test instances
        self.predictions=[]

    def reset(self):
        # set of features for classifier
        self.vocabulary=set()
        # prior probability
        self.prior={}
        # conditional probablility
        self.condProb={}

    def extractVocabulary(self,reviews):
        """
        extract features from training data and store in self.vocabulary.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        for sentiment,review in reviews:
            for token in self.extractReviewTokens(review):
                self.vocabulary.add(token)

    def extractReviewTokens(self,review):
        """
        extract tokens from reviews.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @return: list of strings
        """
        text=[]
        for token in review:
            # check if pos tags are included in review e.g. ("bad","JJ")
            if len(token)==2 and self.discard_closed_class:
                if token[1][0:2] in ["NN","JJ","RB","VB"]: text.append(token)
            else:
                text.append(token)
        if self.bigrams:
            for bigram in ngrams(review,2): text.append(bigram)
        if self.trigrams:
            for trigram in ngrams(review,3): text.append(trigram)
        return text

    def create_vocab_dict(self):
        vocab_to_id = {}
        for word in self.vocabulary:
            vocab_to_id[word] = len(vocab_to_id)
        return vocab_to_id

    def train(self,reviews):
        """
        train NaiveBayesText classifier.

        1. reset self.vocabulary, self.prior and self.condProb
        2. extract vocabulary (i.e. get features for training)
        3. get prior and conditional probability for each label ("POS","NEG") and store in self.prior and self.condProb
           note: to get conditional concatenate all text from reviews in each class and calculate token frequencies
                 to speed this up simply do one run of the movie reviews and count token frequencies if the token is in the vocabulary,
                 then iterate the vocabulary and calculate conditional probability (i.e. don't read the movie reviews in their entirety
                 each time you need to calculate a probability for each token in the vocabulary)

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        # TODO Q1
        self.reset()
        self.extractVocabulary(reviews)
        self.vocab_to_id = self.create_vocab_dict()

        start_count = 0 if not self.smoothing else 1
        positive_vocab = np.array([start_count for _ in range(len(self.vocabulary))])
        negative_vocab = np.array([start_count for _ in range(len(self.vocabulary))])
        sentiment_prior = [0,0]

        for sentiment, review in reviews:
            if sentiment == 'POS':
                vocab = positive_vocab
                sentiment_prior[0] += 1
            elif sentiment == 'NEG':
                vocab = negative_vocab
                sentiment_prior[1] += 1

            for word in review:
                if word in self.vocabulary:
                    text_id = self.vocab_to_id[word]
                    vocab[text_id] += 1

        self.prior = {'POS':sentiment_prior[0], 'NEG':sentiment_prior[1]}
        self.condProb['POS'] = positive_vocab/sum(positive_vocab)
        self.condProb['NEG'] = negative_vocab/sum(negative_vocab)

    def test(self,reviews):
        """
        test NaiveBayesText classifier and store predictions in self.predictions.
        self.predictions should contain a "+" if prediction was correct and "-" otherwise.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        # TODO Q1
        for label, review_text in reviews:
            pred = self.classify_review(review_text)
            if pred == label:
                self.predictions.append('+')
            else:
                self.predictions.append('-')

    def classify_review(self, review_text):
        pos_score = np.log(self.prior['POS'])
        neg_score = np.log(self.prior['NEG'])

        for word in review_text:
            if word in self.vocabulary:
                word_id = self.vocab_to_id[word]
                pos_score += np.log(self.condProb['POS'][word_id])
                neg_score += np.log(self.condProb['NEG'][word_id])

        output = 'POS' if pos_score > neg_score else 'NEG'
        return output


class SVMText(Evaluation):
    def __init__(self,bigrams,trigrams,discard_closed_class):
        """
        initialisation of SVMText object

        @param bigrams: add bigrams?
        @type bigrams: boolean

        @param trigrams: add trigrams?
        @type trigrams: boolean

        @param svmlight_dir: location of smvlight binaries
        @type svmlight_dir: string

        @param svmlight_dir: location of smvlight binaries
        @type svmlight_dir: string

        @param discard_closed_class: restrict unigrams to nouns, adjectives, adverbs and verbs?
        @type discard_closed_class: boolean
        """
        self.svm_classifier = svm.SVC()
        self.predictions=[]
        self.vocabulary=set()
        # add in bigrams?
        self.bigrams=bigrams
        # add in trigrams?
        self.trigrams=trigrams
        # restrict to nouns, adjectives, adverbs and verbs?
        self.discard_closed_class=discard_closed_class

    def create_vocab_dict(self):
        vocab_to_id = {}
        for word in self.vocabulary:
            vocab_to_id[word] = len(vocab_to_id)
        return vocab_to_id

    def extractVocabulary(self,reviews):
        self.vocabulary = set()
        for sentiment, review in reviews:
            for token in self.extractReviewTokens(review):
                 self.vocabulary.add(token)

    def extractReviewTokens(self,review):
        """
        extract tokens from reviews.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @return: list of strings
        """
        text=[]
        for term in review:
            # check if pos tags are included in review e.g. ("bad","JJ")
            if len(term)==2 and self.discard_closed_class:
                if term[1][0:2] in ["NN","JJ","RB","VB"]: text.append(term)
            else:
                text.append(term)
        if self.bigrams:
            for bigram in ngrams(review,2): text.append(term)
        if self.trigrams:
            for trigram in ngrams(review,3): text.append(term)
        return text

    def getFeatures(self,reviews):
        """
        get vectors for svmlight from reviews.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """

        # TODO Q6.
        self.extractVocabulary(reviews)
        self.vocab_to_id = self.create_vocab_dict()

        self.input_features = []
        self.labels = []

        from tqdm import tqdm

        for sentiment, review in tqdm(reviews):
            feature = self.getFeature(review)
            self.input_features.append(feature)
            self.labels.append(sentiment)

    def getFeature(self, review):
        feature = np.array([0 for _ in range(len(self.vocabulary))])
        for word in review:
            if word in self.vocabulary:
                word_id = self.vocab_to_id[word]
                feature[word_id] += 1
        return feature

    def train(self,reviews):
        """
        train svm

        @param train_data: training data
        @type train_data: list of (string, list) tuples corresponding to (label, content)
        """
        # function to determine features in training set.
        self.getFeatures(reviews)
        # function to find vectors (feature, value pairs)

        # train SVM model
        self.svm_classifier.fit(self.input_features, self.labels)

    def test(self,reviews):
        """
        test svm

        @param test_data: test data
        @type test_data: list of (string, list) tuples corresponding to (label, content)
        """

        # TODO Q1
        for label, review_text in reviews:
            pred = self.classify_review(review_text)
            if pred == label:
                self.predictions.append('+')
            else:
                self.predictions.append('-')

    def classify_review(self, review_text):
        feature = self.getFeature(review_text)
        output = self.svm_classifier.predict([feature])
        return output[0]
