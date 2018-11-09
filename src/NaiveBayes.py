import numpy as np
np.random.seed(1234)

from FeatureGenerator import FeatureGenerator
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB


class NaiveBayes:
    """ """

    def __init__(self, features={}, train_split=0.8, distribution="Bernoulli"):
        """ """
        self.Tag = {
            "OTH": 0, "BKG": 1, "CTR": 2, "NA": 7,
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

        self.Cit = {
            "Yes": 0, "No": 1,
        }

        self.Ref = {
            "Self": 0, "Other": 1, "None": 2,
        }


        self.distribution = distribution
        self.train_split = train_split
        self.features = features
        self.feed()


    def feed(self):
        """ """
        papers = self.features.keys()
        order = np.random.permutation(len(papers))

        self.train_papers = []
        for i in range(int(self.train_split*len(papers))):
            self.train_papers.append(papers[order[i]])

        self.test_papers = []
        for i in range(int(self.train_split*len(papers))+1, len(papers)):
            self.test_papers.append(papers[order[i]])

        self.train_X, self.train_y = self.extractFeatures(self.train_papers)
        self.test_X, self.test_y = self.extractFeatures(self.test_papers)


    def extractFeatures(self, papers):
        """ """
        X = []; y = []
        for xmlfile in papers:
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
                    feature.append(self.Cit[eachsentencefeature["Cit"]])
                    feature.append(self.Ref[eachsentencefeature["Ref"]])
                    feature.append(self.Tag[eachsentencefeature["His"]])
                    label = self.Tag[eachsentencefeature["Tag"]]
                except KeyError:
                    continue

                X.append(feature)
                y.append(label)

        X = np.asarray(X)
        y = np.asarray(y)
        return X, y


    def train(self):
        """
        issues:
            - more misclassifications
            - Bernoulli Accuracy: 68.4
            - Multinomial Accuracy: 59.1
            - Complement Accuracy: 49.3
            - Gaussian Accuracy: 22.0

        """
        print("Training on %d papers" %(len(self.train_papers)))
        if self.distribution == "Multinomial": 
            self.mnb = MultinomialNB()
            y_pred = self.mnb.fit(self.train_X, self.train_y).predict(self.train_X)
            print("Number of mislabeled sentences out of a total %d sentences : %d"
                  % (self.train_X.shape[0],(self.train_y != y_pred).sum()))
            print("Train Accuracy: %f"
                % (self.accuracy((self.train_y != y_pred).sum(), self.train_X.shape[0])))

        elif self.distribution == "Gaussian":
            self.gnb = GaussianNB()
            y_pred = self.gnb.fit(self.train_X, self.train_y).predict(self.train_X)
            print("Number of mislabeled sentences out of a total %d sentences : %d"
                  % (self.train_X.shape[0],(self.train_y != y_pred).sum()))
            print("Train Accuracy: %f"
                % (self.accuracy((self.train_y != y_pred).sum(), self.train_X.shape[0])))

        elif self.distribution == "Complement":
            self.cnb = ComplementNB()
            y_pred = self.cnb.fit(self.train_X, self.train_y).predict(self.train_X)
            print("Number of mislabeled sentences out of a total %d sentences : %d"
                  % (self.train_X.shape[0],(self.train_y != y_pred).sum()))
            print("Train Accuracy: %f"
                % (self.accuracy((self.train_y != y_pred).sum(), self.train_X.shape[0])))

        elif self.distribution == "Bernoulli":
            self.bnb = BernoulliNB()
            y_pred = self.bnb.fit(self.train_X, self.train_y).predict(self.train_X)
            print("Number of mislabeled sentences out of a total %d sentences : %d"
                  % (self.train_X.shape[0],(self.train_y != y_pred).sum()))
            print("Train Accuracy: %f"
                % (self.accuracy((self.train_y != y_pred).sum(), self.train_X.shape[0])))                   


    def test(self):
        """ """
        print("Testing on %d papers" %(len(self.test_papers)))
        if self.distribution == "Multinomial": 
            self.mnb = MultinomialNB()
            y_pred = self.mnb.fit(self.test_X, self.test_y).predict(self.test_X)
            print("Number of mislabeled sentences out of a total %d sentences : %d"
                  % (self.test_X.shape[0],(self.test_y != y_pred).sum()))
            print("Test Accuracy: %f"
                % (self.accuracy((self.test_y != y_pred).sum(), self.test_X.shape[0])))

        elif self.distribution == "Gaussian":
            self.gnb = GaussianNB()
            y_pred = self.gnb.fit(self.test_X, self.test_y).predict(self.test_X)
            print("Number of mislabeled sentences out of a total %d sentences : %d"
                  % (self.test_X.shape[0],(self.test_y != y_pred).sum()))
            print("Test Accuracy: %f"
                % (self.accuracy((self.test_y != y_pred).sum(), self.test_X.shape[0])))

        elif self.distribution == "Complement":
            self.cnb = ComplementNB()
            y_pred = self.cnb.fit(self.test_X, self.test_y).predict(self.test_X)
            print("Number of mislabeled sentences out of a total %d sentences : %d"
                  % (self.test_X.shape[0],(self.test_y != y_pred).sum()))
            print("Test Accuracy: %f"
                % (self.accuracy((self.test_y != y_pred).sum(), self.test_X.shape[0])))

        elif self.distribution == "Bernoulli":
            self.bnb = BernoulliNB()
            y_pred = self.bnb.fit(self.test_X, self.test_y).predict(self.test_X)
            print("Number of mislabeled sentences out of a total %d sentences : %d"
                  % (self.test_X.shape[0],(self.test_y != y_pred).sum()))
            print("Test Accuracy: %f"
                % (self.accuracy((self.test_y != y_pred).sum(), self.test_X.shape[0])))


    def accuracy(self, misclassifications, samples):
        return (1-(misclassifications/(samples*1.0)))*100.0


if __name__ == '__main__':
    xmlcorpora = "../data/corpora/AZ_distribution/"
    featureGen = FeatureGenerator(xmlcorpora)
    classifer = NaiveBayes(features=featureGen.features, distribution="Bernoulli", train_split=0.85)
    classifer.train()
    classifer.test()