import os
import xml.etree.ElementTree as ET


class FeaturGenerator:
    """ """
    def __init__(self, xmlcorpora):
        """ """
        self.features = {}
        self.totalpartitions = 20
        self.location = {
            1: "A", 2: "B", 3: "C", 4: "D", 5: "E",
            6: "E", 7: "F", 8: "F", 9: "F", 10: "F", 
            11: "F", 12: "F", 13: "F", 14: "F", 15: "G",
            16: "G", 17: "H", 18: "H", 19: "I", 20: "J",
        }
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

    def getTags(self, xmlfile):
        """ """
        for eachsentence in self.root.iter("A-S"):
            sentenceID = eachsentence.attrib["ID"]
            self.features[xmlfile][sentenceID] = {}
            try:
                self.features[xmlfile][sentenceID]["Tag"] = eachsentence.attrib["AZ"]
            except KeyError:
                self.features[xmlfile][sentenceID]["Tag"] = "NA"

        for eachsentence in self.root.iter("S"):
            sentenceID = eachsentence.attrib["ID"]
            self.features[xmlfile][sentenceID] = {}
            try:
                self.features[xmlfile][sentenceID]["Tag"] = eachsentence.attrib["AZ"]
            except KeyError:
                self.features[xmlfile][sentenceID]["Tag"] = "NA"

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
        for eachpara in self.root.iter("ABSTRACT"):
            total = 1
            for i in eachpara.iter("A-S"):
                total += 1

            line = 1
            for eachsentence in eachpara.iter("A-S"):
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
        
        for eachpara in self.root.iter("P"):
            total = 1
            for i in eachpara.iter("S"):
                total += 1

            line = 1
            for eachsentence in eachpara.iter("S"):
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


if __name__ == "__main__":
    xmlcorpora = "../corpora/AZ_distribution/"
    featureGen = FeaturGenerator(xmlcorpora)