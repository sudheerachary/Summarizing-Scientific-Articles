import os
import numpy as np
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureGenerator:
    """ """

    def __init__(self, xmlcorpora):
        """ 
        issues:
            - add stopwords
        """
        self.top = 30
        self.text = []
        self.features = {}
        self.totalpartitions = 20
        self.location = {
            1: "A", 2: "B", 3: "C", 4: "D", 5: "E",
            6: "E", 7: "F", 8: "F", 9: "F", 10: "F", 
            11: "F", 12: "F", 13: "F", 14: "F", 15: "G",
            16: "G", 17: "H", 18: "H", 19: "I", 20: "J",
        }
        self.prototypes = ["Introduction", "Implementation", "Example", "Conclusion","Result",
                        "Evaluation","Solution","Discussion","Further Work","Data","Related Work",
                        "Experiment","Problems","Method","Problem Statement","Non-Prototypical"]
        self.stopwords = stop_words = set(stopwords.words('english'))
        self.parseXML(xmlcorpora)


    def parseXML(self, xmlcorpora):
        """ """
        for xmlfile in os.listdir(xmlcorpora):
            if ".az-scixml" in xmlfile:
                self.features[xmlfile] = {}
                self.tree = ET.parse(xmlcorpora+xmlfile)
                self.root = self.tree.getroot()
                self.getTags(xmlfile)
                self.absoluteLocation(xmlfile)
                self.explicitStructure(xmlfile)
                self.paragraphStructure(xmlfile)
                self.headlines(xmlfile)
                self.title(xmlfile)
                self.length(xmlfile)
                self.citation(xmlfile)
                self.history(xmlfile)
        self.tfIdf(xmlcorpora)


    def getTags(self, xmlfile):
        """ """
        lines = []
        for eachsentence in self.root.iter("A-S"):
            sentenceID = eachsentence.attrib["ID"]
            self.features[xmlfile][sentenceID] = {}
            try:
                self.features[xmlfile][sentenceID]["Tag"] = eachsentence.attrib["AZ"]
            except KeyError:
                self.features[xmlfile][sentenceID]["Tag"] = "NA"
            lines.append(eachsentence.text)

        for eachsentence in self.root.iter("S"):
            sentenceID = eachsentence.attrib["ID"]
            self.features[xmlfile][sentenceID] = {}
            try:
                self.features[xmlfile][sentenceID]["Tag"] = eachsentence.attrib["AZ"]
            except KeyError:
                self.features[xmlfile][sentenceID]["Tag"] = "NA"
            lines.append(eachsentence.text)

        self.text.append(" ".join(lines))


    def absoluteLocation(self, xmlfile):
        """ """
        for eachsentence in self.root.iter("A-S"):
            sentenceID = eachsentence.attrib["ID"]
            self.features[xmlfile][sentenceID]["Text"] = eachsentence.text

        for eachsentence in self.root.iter("S"):
            sentenceID = eachsentence.attrib["ID"]
            self.features[xmlfile][sentenceID]["Text"] = eachsentence.text            

        ids = self.features[xmlfile].keys()
        eachpartition = len(ids)/self.totalpartitions
        for i in range(len(ids)):
            if i/eachpartition < self.totalpartitions:
                segment = i/eachpartition +1
            else:
                segment = self.totalpartitions
            self.features[xmlfile][ids[i]]["Loc"] = self.location[segment]


    def explicitStructure(self, xmlfile):
        """ """
        for eachsection in self.root.iter("DIV"):
            total = 1
            for i in eachsection.iter("S"):
                total += 1

            line = 1
            for eachsentence in eachsection.iter("S"):
                sentenceID = eachsentence.attrib["ID"]
                if line == 1:
                    self.features[xmlfile][sentenceID]["Sec"] = "FIRST"
                elif line == 2:
                    self.features[xmlfile][sentenceID]["Sec"] = "SECOND"
                elif line == 3:
                    self.features[xmlfile][sentenceID]["Sec"] = "THIRD"
                elif line == total-1:
                    self.features[xmlfile][sentenceID]["Sec"] = "LAST"
                elif line == total-2:
                    self.features[xmlfile][sentenceID]["Sec"] = "SECOND-LAST"
                elif line == total-3: 
                    self.features[xmlfile][sentenceID]["Sec"] = "THIRD-LAST"
                else:
                    self.features[xmlfile][sentenceID]["Sec"] = "SOMEWHERE"
                line += 1

        for abstract in self.root.iter("ABSTRACT"):
            total = 1
            for i in abstract.iter("A-S"):
                total += 1

            line = 1
            for eachsentence in abstract.iter("A-S"):
                sentenceID = eachsentence.attrib["ID"]
                if line == 1:
                    self.features[xmlfile][sentenceID]["Sec"] = "FIRST"
                elif line == 2:
                    self.features[xmlfile][sentenceID]["Sec"] = "SECOND"
                elif line == 3:
                    self.features[xmlfile][sentenceID]["Sec"] = "THIRD"
                elif line == total-1:
                    self.features[xmlfile][sentenceID]["Sec"] = "LAST"
                elif line == total-2:
                    self.features[xmlfile][sentenceID]["Sec"] = "SECOND-LAST"
                elif line == total-3: 
                    self.features[xmlfile][sentenceID]["Sec"] = "THIRD-LAST"
                else:
                    self.features[xmlfile][sentenceID]["Sec"] = "SOMEWHERE"
                line += 1                


    def paragraphStructure(self, xmlfile):
        """ """
        for abstract in self.root.iter("ABSTRACT"):
            total = 1
            for i in abstract.iter("A-S"):
                total += 1

            line = 1
            for eachsentence in abstract.iter("A-S"):
                sentenceID = eachsentence.attrib["ID"]
                if line == 1:
                    self.features[xmlfile][sentenceID]["Par"] = "INITIAL"
                elif line == total-1:
                    self.features[xmlfile][sentenceID]["Par"] = "FINAL"
                else:
                    self.features[xmlfile][sentenceID]["Par"] = "MEDIAL"
                line += 1                

        for eachpara in self.root.iter("P"):
            total = 1
            for i in eachpara.iter("S"):
                total += 1

            line = 1
            for eachsentence in eachpara.iter("S"):
                sentenceID = eachsentence.attrib["ID"]
                if line == 1:
                    self.features[xmlfile][sentenceID]["Par"] = "INITIAL"
                elif line == total-1:
                    self.features[xmlfile][sentenceID]["Par"] = "FINAL"
                else:
                    self.features[xmlfile][sentenceID]["Par"] = "MEDIAL"
                line += 1


    def headlines(self, xmlfile):
        """ 
        issues:
            - fix headline for abstract section
            - fix exact match with partial match
              after processing headline of the 
              section.
        """
        for eachsection in self.root.iter("DIV"):
            prototype = self.prototypes[-1]
            for header in eachsection.iter("HEADER"):
                if header.text.strip() in self.prototypes:
                    prototype = header.text.strip()
                else:
                    prototype = self.prototypes[-1]
                break

            for eachsentence in eachsection.iter("S"):
                sentenceID = eachsentence.attrib["ID"]
                self.features[xmlfile][sentenceID]["Head"] = prototype

        for eachsentence in self.root.iter("A-S"):
            sentenceID =  eachsentence.attrib["ID"]
            self.features[xmlfile][sentenceID]["Head"] = "Introduction"


    def length(self, xmlfile):
        """
        issues:
            - fix length without stopwords
        """
        for eachsentence in self.root.iter("A-S"):
            sentenceID = eachsentence.attrib["ID"]
            length = len(eachsentence.text.split())
            self.features[xmlfile][sentenceID]["Len"] = length

        for eachsentence in self.root.iter("S"):
            sentenceID = eachsentence.attrib["ID"]
            length = len(eachsentence.text.split())
            self.features[xmlfile][sentenceID]["Len"] = length            


    def title(self, xmlfile):
        """ 
        issues:
            - fix exact match of title words with sentence words
            - fix stopword removal, stemmer, parser
        """
        title = None
        for tag in self.root.iter("TITLE"):
            title = [w for w in tag.text.split() if not w in self.stopwords]

        for eachsentence in self.root.iter("A-S"):
            sentenceID = eachsentence.attrib["ID"]
            self.features[xmlfile][sentenceID]["Title"] = "No"
            for w in eachsentence.text.split():
                if not w in self.stopwords and w in title:
                    self.features[xmlfile][sentenceID]["Title"] = "Yes"
                    break
        
        for eachsentence in self.root.iter("S"):
            sentenceID = eachsentence.attrib["ID"]
            self.features[xmlfile][sentenceID]["Title"] = "No"
            for w in eachsentence.text.split():
                if not w in self.stopwords and w in title:
                    self.features[xmlfile][sentenceID]["Title"] = "Yes"
                    break


    def citation(self, xmlfile):
        """
        issues:
            - fix stop words, years
            - add positional information to feature 
        """
        authors = []
        for author in self.root.iter("AUTHOR"):
            authors.extend(author.text.split())

        for eachsentence in self.root.iter("A-S"):
            sentenceID = eachsentence.attrib["ID"]
            self.features[xmlfile][sentenceID]["Ref"] = "None"
            self.features[xmlfile][sentenceID]["Cit"] = "No"
            for ref in eachsentence.iter("REF"):
                self.features[xmlfile][sentenceID]["Ref"] = "Other"
                self.features[xmlfile][sentenceID]["Cit"] = "Yes"
                for author in ref.text.split():
                    if author in authors:
                        self.features[xmlfile][sentenceID]["Ref"] = "Self"
                        break

        for eachsentence in self.root.iter("S"):
            sentenceID = eachsentence.attrib["ID"]
            self.features[xmlfile][sentenceID]["Ref"] = "None"
            self.features[xmlfile][sentenceID]["Cit"] = "No"
            for ref in eachsentence.iter("REF"):
                self.features[xmlfile][sentenceID]["Ref"] = "Other"
                self.features[xmlfile][sentenceID]["Cit"] = "Yes"
                for author in ref.text.split():
                    if author in authors:
                        self.features[xmlfile][sentenceID]["Ref"] = "Self"
                        break


    def history(self, xmlfile):
        """
        issues:
            - begining of the paper 
        """
        prevTag = "NA"
        for eachsentence in self.root.iter("A-S"):
            sentenceID = eachsentence.attrib["ID"]
            self.features[xmlfile][sentenceID]["His"] = prevTag
            prevTag = self.features[xmlfile][sentenceID]["Tag"]

        for eachsentence in self.root.iter("S"):
            sentenceID = eachsentence.attrib["ID"]
            self.features[xmlfile][sentenceID]["His"] = prevTag
            prevTag = self.features[xmlfile][sentenceID]["Tag"]


    def tfIdf(self, xmlcorpora):
        """ """
        vectorizer = TfidfVectorizer(input='content', encoding='utf-8', decode_error='strict', 
            lowercase=True, preprocessor=None, tokenizer=None, analyzer='word', stop_words=None, 
            ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, norm='l2', 
            use_idf=True, smooth_idf=True, sublinear_tf=False)
        vectorizer.fit_transform(self.text)
        indices = np.argsort(vectorizer.idf_)[::-1]
        features = vectorizer.get_feature_names()
        self.tfidf = [features[i] for i in indices[:self.top]]

        for xmlfile in os.listdir(xmlcorpora):
            if ".az-scixml" in xmlfile:
                root = ET.parse(xmlcorpora+xmlfile)
                for eachsentence in root.iter("A-S"):
                    sentenceID = eachsentence.attrib["ID"]
                    self.features[xmlfile][sentenceID]["TfIdf"] = "NO"
                    for w in eachsentence.text.split():
                        if w in self.tfidf:
                            self.features[xmlfile][sentenceID]["TfIdf"] = "YES"
                            break

                for eachsentence in root.iter("S"):
                    sentenceID = eachsentence.attrib["ID"]
                    self.features[xmlfile][sentenceID]["TfIdf"] = "NO"
                    for w in eachsentence.text.split():
                        if w in self.tfidf:
                            self.features[xmlfile][sentenceID]["TfIdf"] = "YES"
                            break


if __name__ == "__main__":
    
    # locate corpus data 
    xmlcorpora = "../data/corpora/AZ_distribution/"

    # generate gold-standard features
    featureGen = FeatureGenerator(xmlcorpora)
    
    # sample feature vectors
    print featureGen.features["9405001.az-scixml"]["S-8"]
    print featureGen.features["9405001.az-scixml"]["A-0"]