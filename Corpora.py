import os, codecs, sys
from nltk.stem.porter import PorterStemmer

class MovieReviewCorpus():
    def __init__(self,stemming,pos):
        """
        initialisation of movie review corpus.

        @param stemming: use porter's stemming?
        @type stemming: boolean

        @param pos: use pos tagging?
        @type pos: boolean
        """
        # raw movie reviews
        self.reviews=[]
        # held-out train/test set
        self.train=[]
        self.test=[]
        # folds for cross-validation
        self.folds={}
        # porter stemmer
        self.stemmer=PorterStemmer() if stemming else None
        # part-of-speech tags
        self.pos=pos
        # import movie reviews
        self.get_reviews()

    def get_reviews(self):
        """
        processing of movie reviews.

        1. parse reviews in data/reviews and store in self.reviews.

           the format expected for reviews is: [(string,list), ...] e.g. [("POS",["a","good","movie"]), ("NEG",["a","bad","movie"])].
           in data/reviews there are .tag and .txt files. The .txt files contain the raw reviews and .tag files contain tokenized and pos-tagged reviews.

           to save effort, we recommend you use the .tag files. you can disregard the pos tags to begin with and include them later.
           when storing the pos tags, please use the format for each review: ("POS/NEG", [(token, pos-tag), ...]) e.g. [("POS",[("a","DT"), ("good","JJ"), ...])]

           to use the stemmer the command is: self.stemmer.stem(token)

        2. store training and held-out reviews in self.train/test. files beginning with cv9 go in self.test and others in self.train

        3. store reviews in self.folds. self.folds is a dictionary with the format: self.folds[fold_number] where fold_number is an int 0-9.
           you can get the fold number from the review file name.
        """
        # TODO Q0
        pos_path = 'data/reviews/POS'
        pos_files = [f'{pos_path}/{file_path}' for file_path in os.listdir(pos_path) if file_path.split('.')[-1] == 'tag']
        pos_reviews = [(path[19], 'POS', self.read_tag_file(path)) for path in pos_files]

        neg_path = 'data/reviews/NEG'
        neg_files = [f'{neg_path}/{file_path}' for file_path in os.listdir(neg_path) if file_path.split('.')[-1] == 'tag']
        neg_reviews = [(path[19], 'NEG', self.read_tag_file(path)) for path in neg_files]

        folds = {k:[] for k in range(10)}

        for fold, label, text in pos_reviews + neg_reviews:
            folds[int(fold)].append((label, text))

        self.train = [review for i in range(9) for review in folds[i]]
        self.test = folds[9]
        self.reviews = self.train + self.test
        self.folds = folds

    def read_tag_file(self, path):
        review_text = []
        with open(path, 'r') as f:
            for line in f:
                word = line.split()
                if len(word)==2:
                    if self.stemmer:    word[0] = self.stemmer.stem(word[0])
                    if self.pos:        word = tuple(word)
                    else:               word = word[0]
                    review_text.append(word)
        return review_text
