import itertools
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234)

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from nltk.corpus import stopwords 
from FeatureGenerator import FeatureGenerator

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Conv1D, Dense, Input, Flatten, Dropout, BatchNormalization
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

class NaiveBayes:
    """ """

    def __init__(self, features={}, train_split=0.8, distribution="Bernoulli"):
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
        self.Cit = {
            "Yes": 0, "No": 1,
        }
        self.Ref = {
            "Self": 0, "Other": 1, "None": 2,
        }
        self.batch_size = 128; self.epochs = 100;
        self.stop_words = set(stopwords.words('english'))
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
        self.deep_train_X, self.deep_train_y = self.word2Vec(self.train_papers)
        self.deep_train_y = to_categorical(self.deep_train_y)
        self.deep_test_X, self.deep_test_y = self.word2Vec(self.test_papers)
        
    def word2Vec(self, papers):
        """ """
        X = []; y = []
        sequences = []
        filename = '../models/glove.6B.50d.txt.word2vec'
        self.model = KeyedVectors.load_word2vec_format(filename, binary=False)
        for xmlfile in papers:
            for sentenceID in self.features[xmlfile].keys():
                sequence = []
                eachsentence = self.features[xmlfile][sentenceID]
                try:
                    for w in text_to_word_sequence(eachsentence["Text"]):
                            if not w in self.stop_words:
                                sequence.extend(self.model[w])
                    y.append(self.Tag[eachsentence["Tag"]])
                except KeyError:
                    continue
                sequences.append(sequence)
        X = pad_sequences(sequences, maxlen=2000, dtype='float64', padding='post', truncating='post', value=0.0)
        y = np.asarray(y)
        return X, y
    
    def deepCNN(self):
        """ """
        inputs = Input(shape=(2000, 1))
        x = Conv1D(128, 3, strides=1, padding='same', activation="relu")(inputs)
        x = BatchNormalization()(x)
        x = Conv1D(256, 5, strides=1, padding='same', activation="tanh")(x)
        x = Dropout(0.25)(x)
        x = Conv1D(256, 3, strides=1, padding='same', activation="tanh")(x)
        x = Flatten()(x)
        x = Dense(64, activation="tanh")(x)
        x = BatchNormalization()(x)
        x = Dense(32, activation="sigmoid")(x)
        x = Dense(7, activation="sigmoid")(x)
        model = Model(inputs=[inputs], outputs=[x])
        model.compile(optimizer="adamax", loss=["categorical_crossentropy"], metrics=["accuracy"])
        model.summary()
        return model
        
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
            self.clf = MultinomialNB()
            self.clf.fit(self.train_X, self.train_y)
        elif self.distribution == "Gaussian":
            self.clf = GaussianNB()
            self.clf.fit(self.train_X, self.train_y)
        elif self.distribution == "Complement":
            self.clf = ComplementNB()
            self.clf.fit(self.train_X, self.train_y)
        elif self.distribution == "Bernoulli":
            self.clf = BernoulliNB()
            self.clf.fit(self.train_X, self.train_y)
        elif self.distribution == "Deep":
            self.clf = self.deepCNN()
            callback = ModelCheckpoint("../models/weights.{epoch:02d}-{val_acc:.5f}.hdf5", monitor='val_acc', 
                                       verbose=0, save_best_only=True, save_weights_only=True, mode='max', period=1)
            self.clf.fit(x=np.expand_dims(self.deep_train_X, axis=2), y=self.deep_train_y, batch_size=self.batch_size, 
                         epochs=self.epochs, validation_split=0.1, callbacks=[callback])
    

    def test(self):
        """ """
        print("=== Testing on %d papers ===" %(len(self.test_papers)))
        if self.distribution == "Deep":
            y_pred = np.argmax(self.clf.predict(np.expand_dims(self.deep_test_X, axis=2)), axis=1)
            plt.hist(y_pred, density=True)
            plt.savefig('histogram.png')
            return (self.getConfusionMatrix(self.deep_test_y, y_pred),
                self.getPrecisionRecallF1Score(self.deep_test_y, y_pred),
                self.getAccuracy(self.deep_test_y, y_pred))
        else:
            y_pred = self.clf.predict(self.test_X)
            plt.hist(y_pred, density=True)
            plt.savefig('histogram.png')
            return (self.getConfusionMatrix(self.test_y, y_pred),
                self.getPrecisionRecallF1Score(self.test_y, y_pred),
                self.getAccuracy(self.test_y, y_pred))
    

    def getSummary(self, xmlfile):
        """ """
        summary = []
        for sentenceID in self.features[xmlfile].keys():
            feature = []
            eachsentencefeature = self.features[xmlfile][sentenceID]
            if self.distribution == "Deep":
                try:
                    for w in text_to_word_sequence(eachsentencefeature["Text"]):
                            if not w in self.stop_words:
                                feature.extend(self.model[w])
                except KeyError:
                    continue
                X = pad_sequences([feature], maxlen=2000, dtype='float64', padding='post', truncating='post', value=0.0)
                sentenceTag = np.argmax(self.clf.predict(np.expand_dims(X, axis=2)), axis=1)
            else:
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
                sentenceTag = self.clf.predict([feature])[0]
            if sentenceTag in [1, 2, 3, 5]:
                summary.append(eachsentencefeature["Text"])
        print("=== Summary of "+str(xmlfile)+" ===")
        for line in summary:
            print(line)


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
        plt.savefig('confusion_matrix.png')

    def getConfusionMatrix(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred)


    def getPrecisionRecallF1Score(self, y_true, y_pred):
        return precision_recall_fscore_support(y_true, y_pred, average='micro')


    def getAccuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)


if __name__ == '__main__':
    
    # locate corpus data
    xmlcorpora = "../data/corpora/AZ_distribution/"
    
    # generate gold-standard feature vectors
    featureGen = FeatureGenerator(xmlcorpora)

    # initialise and train NB model
    classifer = NaiveBayes(features=featureGen.features, distribution="Bernoulli", train_split=0.8)
    classifer.train()

    # Analysis
    confusionMatrix, precisionRecallF1, accuracy = classifer.test()
    classifer.plotConfusionMatrix(confusionMatrix, range(8))
    print("=== Accuracy: %f ===" %(accuracy))

    # generate summary
    classifer.getSummary('9405001.az-scixml')