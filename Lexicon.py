from Analysis import Evaluation

class SentimentLexicon(Evaluation):
    def __init__(self):
        """
        read in lexicon database and store in self.lexicon
        """
        # if multiple entries take last entry by default
        self.lexicon = self.get_lexicon_dict()
        self.polarity_dict = {'negative':-1, 'positive':1, 'neutral':0, 'both':0}
        self.magnitude_dict = {'weaksubj':1, 'strongsubj':2}

    def get_lexicon_dict(self):
        lexicon_dict = {}
        with open('data/sent_lexicon', 'r') as f:
            for line in f:
                word = line.split()[2].split("=")[1]
                polarity = line.split()[5].split("=")[1]
                magnitude = line.split()[0].split("=")[1]
                lexicon_dict[word] = [magnitude, polarity]
        return lexicon_dict

    def classify(self,reviews,threshold,magnitude):
        """
        classify movie reviews using self.lexicon.
        self.lexicon is a dictionary of word: [polarity_info, magnitude_info], e.g. "bad": ["priorpolarity=negative","type=strongsubj"].
        explore data/sent_lexicon to get a better understanding of the sentiment lexicon.
        store the predictions in self.predictions as a list of strings where "+" and "-" are correct/incorrect classifications respectively e.g. ["+","-","+",...]

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @param threshold: threshold to center decisions on. instead of using 0, there may be a bias in the reviews themselves which could be accounted for.
                          experiment for good threshold values.
        @type threshold: integer

        @type magnitude: use magnitude information from self.lexicon?
        @param magnitude: boolean
        """
        # reset predictions
        self.predictions=[]
        # TODO Q0

        for label, review_text in reviews:
            pred = self.classify_review(review_text, threshold, magnitude)
            if pred == label:
                self.predictions.append('+')
            else:
                self.predictions.append('-')

    def classify_review(self, review_text, threshold, magnitude_enabled):
        score = 0
        for word in review_text:
            if word in self.lexicon:
                magnitude, polarity = self.lexicon[word]

                if magnitude_enabled:
                    score += self.magnitude_dict[magnitude] * self.polarity_dict[polarity]
                else:
                    score += self.polarity_dict[polarity]

        output = 'POS' if score > threshold else 'NEG'
        return output
