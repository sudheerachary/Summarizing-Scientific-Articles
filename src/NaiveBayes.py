import itertools
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234)

from FeatureGenerator import FeatureGenerator
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


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
        """ """
        print("=== Training on %d papers ===" %(len(self.train_papers)))
        if self.distribution == "Multinomial": 
            self.mnb = MultinomialNB()
            y_pred = self.mnb.fit(self.train_X, self.train_y).predict(self.train_X)
        elif self.distribution == "Gaussian":
            self.gnb = GaussianNB()
            y_pred = self.gnb.fit(self.train_X, self.train_y).predict(self.train_X)
        elif self.distribution == "Complement":
            self.cnb = ComplementNB()
            y_pred = self.cnb.fit(self.train_X, self.train_y).predict(self.train_X)
        elif self.distribution == "Bernoulli":
            self.bnb = BernoulliNB()
            y_pred = self.bnb.fit(self.train_X, self.train_y).predict(self.train_X)

    def test(self):
        """ """
        print("=== Testing on %d papers ===" %(len(self.test_papers)))
        if self.distribution == "Multinomial": 
            self.mnb = MultinomialNB()
            y_pred = self.mnb.fit(self.test_X, self.test_y).predict(self.test_X)
        elif self.distribution == "Gaussian":
            self.gnb = GaussianNB()
            y_pred = self.gnb.fit(self.test_X, self.test_y).predict(self.test_X)
        elif self.distribution == "Complement":
            self.cnb = ComplementNB()
            y_pred = self.cnb.fit(self.test_X, self.test_y).predict(self.test_X)
        elif self.distribution == "Bernoulli":
            self.bnb = BernoulliNB()
            y_pred = self.bnb.fit(self.test_X, self.test_y).predict(self.test_X)
        return (self.getConfusionMatrix(self.test_y, y_pred),
            self.getPrecisionRecallF1Score(self.test_y, y_pred),
            self.getAccuracy(self.test_y, y_pred))
    

    def plotConfusionMatrix(self, cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

    def getConfusionMatrix(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred)


    def getPrecisionRecallF1Score(self, y_true, y_pred):
        return precision_recall_fscore_support(y_true, y_pred, average='micro')


    def getAccuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)


if __name__ == '__main__':
    xmlcorpora = "../data/corpora/AZ_distribution/"
    featureGen = FeatureGenerator(xmlcorpora)
    classifer = NaiveBayes(features=featureGen.features, distribution="Bernoulli", train_split=0.85)
    classifer.train()
    confusionMatrix, precisionRecallF1, accuracy = classifer.test()
    classifer.plotConfusionMatrix(confusionMatrix, range(8))
    print("=== Accuracy: %f ===" %(accuracy))