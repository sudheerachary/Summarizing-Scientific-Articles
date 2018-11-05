import os
import xml.etree.ElementTree as ET


class FeaturGenerator:
    """ """
    def __init__(self, xmlcorpora):
        """ """
        self.features = {}
        self.totalpartitions = 20
        self.xmlcorpora = xmlcorpora
        self.location = {
            1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E',
            6: 'E', 7: 'F', 8: 'F', 9: 'F', 10: 'F', 
            11: 'F', 12: 'F', 13: 'F', 14: 'F', 15: 'G',
            16: 'G', 17: 'H', 18: 'H', 19: 'I', 20: 'J',
        }
        self.parseXML()

    def parseXML(self):
        """ """
        for xmlfile in os.listdir(self.xmlcorpora):
            if ".az-scixml" in xmlfile:
                self.features[xmlfile] = {}
                self.tree = ET.parse(self.xmlcorpora+xmlfile)
                self.root = self.tree.getroot()
                self.getTags(xmlfile)
                self.absoluteLocation(xmlfile)

    def getTags(self, xmlfile):
        """ """
        self.features[xmlfile]['Tag'] = []
        for eachsentence in self.root.iter('A-S'):
            try:
                self.features[xmlfile]['Tag'].append(eachsentence.attrib['AZ'])
            except KeyError:
                self.features[xmlfile]['Tag'].append('NA')

        for eachsentence in self.root.iter('S'):
            try:
                self.features[xmlfile]['Tag'].append(eachsentence.attrib['AZ'])
            except KeyError:
                self.features[xmlfile]['Tag'].append('NA')

    def absoluteLocation(self, xmlfile):
        """ """
        self.lines = []
        for eachsentence in self.root.iter('A-S'):
            self.lines.append(eachsentence.text)

        for eachsentence in self.root.iter('S'):
            self.lines.append(eachsentence.text)            

        self.features[xmlfile]['Loc'] = []
        eachpartition = len(self.lines)/self.totalpartitions
        for i in range(len(self.lines)):
            if i/eachpartition < self.totalpartitions:
                segment = i/eachpartition +1
            else:
                segment = self.totalpartitions
            self.features[xmlfile]['Loc'].append(self.location[segment])

if __name__ == '__main__':
    xmlcorpora = '../corpora/AZ_distribution/'
    featureGen = FeaturGenerator(xmlcorpora)