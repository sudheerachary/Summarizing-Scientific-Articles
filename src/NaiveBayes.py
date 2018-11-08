import numpy as np
from FeatureGenerator import FeatureGenerator
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB

class NaiveBayes:
    """ """

    def __init__(self, features):
        """ """
        self.Tag = {
            "OTH": 0, "BKG": 1, "CTR": 2,
            "AIM": 3, "OWN": 4, "BAS": 5, "TXT": 6,
        }

        self.Loc = {
            "A": 0, "B": 1, "C": 2, "D": 3, "E": 4,
            "F": 5, "G": 6, "H": 7, "I": 8, "J": 9,
        }

        self.Par = {
            "INITIAL": 0, "MEDIAL": 1, "FINAL": 2,
        }

        self.Title = {
            "Yes": 0, "No": 1,
        }

        self.Head = {
            "Introduction": 0, "Implementation": 1, "Example": 2, "Conclusion": 3, "Result": 4,
            "Evaluation": 5, "Solution": 6, "Discussion": 7, "Further Work": 8, "Data": 9,
            "Related Work": 10, "Experiment": 11, "Problems": 12, "Method": 13, 
            "Problem Statement": 14, "Non-Prototypical": 15,
        }

        self.TfIdf = {
            "YES": 0, "NO": 1,
        }

        self.Sec = {
            "FIRST": 0, "SECOND": 1, "THIRD": 2, "LAST": 3,
            "SECOND-LAST": 4, "THIRD-LAST": 5,"SOMEWHERE": 6,
        }
        self.features = features
        self.feed()

    def feed(self):
        """ """
        self.X = []; self.y = []
        for xmlfile in self.features.keys():
            xmlfile = self.features[xmlfile]
            for eachsentenceID in xmlfile.keys():
                feature = []
                eachsentencefeature = xmlfile[eachsentenceID]
                
                try:
                    feature.append(eachsentencefeature["Len"])
                    feature.append(self.Loc[eachsentencefeature["Loc"]])
                    feature.append(self.Par[eachsentencefeature["Par"]])
                    feature.append(self.Sec[eachsentencefeature["Sec"]])     
                    feature.append(self.Title[eachsentencefeature["Title"]])
                    feature.append(self.Head[eachsentencefeature["Head"]])
                    feature.append(self.TfIdf[eachsentencefeature["TfIdf"]])
                    label = self.Tag[eachsentencefeature["Tag"]]
                except KeyError:
                    continue

                self.X.append(feature)
                self.y.append(label)

        self.X = np.asarray(self.X)
        self.y = np.asarray(self.y)


    def train(self, distribution="Multinomial"):
        """
        issues:
            - more misclassifications
            - Bernoulli Accuracy: 68.4
            - Multinomial Accuracy: 59.1
            - Complement Accuracy: 49.3
            - Gaussian Accuracy: 22.0

        """
        if distribution == "Multinomial": 
            self.mnb = MultinomialNB()
            y_pred = self.mnb.fit(self.X, self.y).predict(self.X)
            print("Number of mislabeled sentences out of a total %d sentences : %d"
                  % (self.X.shape[0],(self.y != y_pred).sum()))
            print("Accuracy: %f"
                % (self.accuracy((self.y != y_pred).sum(), self.X.shape[0])))

        elif distribution == "Gaussian":
            self.gnb = GaussianNB()
            y_pred = self.gnb.fit(self.X, self.y).predict(self.X)
            print("Number of mislabeled sentences out of a total %d sentences : %d"
                  % (self.X.shape[0],(self.y != y_pred).sum()))
            print("Accuracy: %f"
                % (self.accuracy((self.y != y_pred).sum(), self.X.shape[0])))

        elif distribution == "Complement":
            self.cnb = ComplementNB()
            y_pred = self.cnb.fit(self.X, self.y).predict(self.X)
            print("Number of mislabeled sentences out of a total %d sentences : %d"
                  % (self.X.shape[0],(self.y != y_pred).sum()))
            print("Accuracy: %f"
                % (self.accuracy((self.y != y_pred).sum(), self.X.shape[0])))

        elif distribution == "Bernoulli":
            self.bnb = BernoulliNB()
            y_pred = self.bnb.fit(self.X, self.y).predict(self.X)
            print("Number of mislabeled sentences out of a total %d sentences : %d"
                  % (self.X.shape[0],(self.y != y_pred).sum()))
            print("Accuracy: %f"
                % (self.accuracy((self.y != y_pred).sum(), self.X.shape[0])))                   


    def test(self):
        """ """
        pass


    def accuracy(self, misclassifications, samples):
        return (1-(misclassifications/(samples*1.0)))*100.0


if __name__ == '__main__':
    xmlcorpora = "../data/corpora/AZ_distribution/"
    featureGen = FeatureGenerator(xmlcorpora)
    classifer = NaiveBayes(featureGen.features)
    classifer.train(distribution="Bernoulli")