# Summarizing-Scientific-Articles

## FeatureGenerator.py
  ```python
  from FeatureGenerator import FeatureGenerator
  
  # locate corpus data 
  xmlcorpora = "../data/corpora/AZ_distribution/"

  # generate gold-standard features
  featureGen = FeatureGenerator(xmlcorpora)

  # sample feature vectors
  # featureGen.features[<xmlfile_name>][<line_ID>]
  
  # example
  print featureGen.features["9405001.az-scixml"]["A-0"]
  ```
  - generates features from parsing XML paper format
  - each sentence has 14 features along with a tag, stored in `self.features` in FeatureGenerator object
  - Implemented features capture
    - *absolute location*
    - *Tf-Idf scoring*
    - *history*
    - *citations & references*
    - *title & length*
    - *explicit structure*

## NaiveBayes.py
  ```python
  from FeatureGenerator import FeatureGenerator
  from NaiveBayes import NaiveBayes
  
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
  ```
  - gets the feature vectors of each sentence from FeatureGenerator object
  - trains with different classifer as choosen `distribution="Bernoulli"`, available options include
    - *Compliment*
    - *Bernoulli*
    - *Multinomial*
    - *Gaussian*
    - *Deep*
  
  - Implemented classifiers
    - *Gaussian Naive Bayes* **(acc: 25%)**
    - *Multinomial Naive Bayes* **(acc: 72%)**
    - *Bernoulli Naive Bayes* **(acc: 80%)**
    - *Compliment Naive Bayes* **(acc: 73%)**
    - *Deep CNN Network* **(acc: 71%)**

## clean.py
  ```
  python clean.py inputFile | awk '!uniq[substr($0, 0, 10000)]++' > outFile
  ```
  - Basic cleaning of the `inputFile` and writing to `outFile`, each line in `outFile` being a sentence from the `inputFile`.
  - May have errors, ignore them or manually correct if you want.

## script.py
  ```
  python script.py inputFile outFile annotationFile
  ```
  - Send the `inputFile` (output of `clean.py`).
  - Each line from the `inputFile` pops up and you will be asked to tag it.
  - The details of the key for each tag will be output under each sentence.
  - Enter the corresponding tag and that line will be appended to `outFile` and your tag to the `annotationFile`.
  - Used RECHECK(R) tag incase you are not sure about it. Do it manually later, etc.
  - Used IGNORE(I) tag incase that sentence is trash and is not cleaned well from `clean.py` file.
  - The line and the tag wont be written incase of IGNORE tag.
