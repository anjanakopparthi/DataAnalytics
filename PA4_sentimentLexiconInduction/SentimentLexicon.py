import re
import os
import sys
import math
import getopt
import operator


class SentimentLexicon:

  class TrainSplit:
    """Split the dataset into train and test.  
    """
    def __init__(self):
      self.train = []
      self.test  = []

  class Example:
    """Every document has a label which is denoted by klass either 'pos' or 'neg'.
       words is a list of strings/sentences in the text file.
    """
    def __init__(self):
      self.klass = ''
      self.words = [] 

  #----------------------------------------------------
  def __init__(self):
    
    """initializations"""   
    
    self.kFolds = 10
    self.near = 20
    self.posSeed = 'excellent'
    self.negSeed = 'poor'

    # Extra Question-1
    #self.negSeed = 'worst'
    #self.posSeed = 'good'
    # self.negSeed = 'bad'
    # self.posSeed = 'great'
    
    self.negHits = {}
    self.posHits = {}
    self.posSeedCount = 0
    self.negSeedCount = 0

  #----------------------------------------------------
  def generate_patterns(self, words):
    """
    The following codes extracts the phrases following the Turneys paper using regular expressions
    and returns a list of sentiment phrases including the POS tagger
    """
    # regex patterns are the following 
    pattern_matches = []
    patterns = []

    pattern1 = '''[a-zA-Z0-9/\-']+_JJ_[a-zA-Z0-9/\-]+ [a-zA-Z0-9/\-']+_NNS?_[a-zA-Z0-9/\-]+'''
    patterns.append(pattern1)
                                                   
    pattern2 = '''[a-zA-Z0-9/\-']+_RB[RS]?_[a-zA-Z0-9/\-]+ [a-zA-Z0-9/\-']+_JJ_[a-zA-Z0-9/\-]+ [a-zA-Z0-9/\-\."'!\?:]+_[^(NN|NNS)]_[a-zA-Z0-9/\-]+'''
    patterns.append(pattern2)
                                                  
    pattern3 = '''[a-zA-Z0-9/\-']+_JJ_[a-zA-Z0-9/\-]+ [a-zA-Z0-9/\-']+_JJ_[a-zA-Z0-9/\-]+ [a-zA-Z0-9/\-\."'!\?:]+_[^(NN|NNS)]_[a-zA-Z0-9/\-]+'''
    patterns.append(pattern3)
                                                   
    pattern4 = '''[a-zA-Z0-9/\-']+_NN[S]?_[a-zA-Z0-9/\-]+ [a-zA-Z0-9/\-']+_JJ_[a-zA-Z0-9/\-]+ [a-zA-Z0-9/\-\."'!\?:]+_[^(NN|NNS)]_[a-zA-Z0-9/\-]+'''
    patterns.append(pattern4)
                                                   
    pattern5 = '''[a-zA-Z0-9/\-']+_RB[RS]?_[a-zA-Z0-9/\-]+ [a-zA-Z0-9/\-']+_VB[DNG]?_[a-zA-Z0-9/\-]+'''
    patterns.append(pattern5)

    for sentence in words: 
      for idx, pattern in enumerate(patterns):
        regex_step = re.findall(pattern, sentence)
        for reg in regex_step:
          result = reg.strip()
          if idx==0 or idx == 4:
            pattern_matches.append(result)
            # To check the patterns for rule 1 and rule 5
            # if idx==0:
            #     print("pattern for rule 1 -----> "+result)
            # if idx==4:
            #     print("pattern for rule 5 -----> "+result)
          else:
            result = result.split()
            result = ' '.join(result[:2])
            pattern_matches.append(result)
            # To check the patterns for rule2, rule3 and rule 4
            # if idx==1:
            #     print("pattern for rule 2 -----> "+result)
            # elif idx==2:
            #     print("pattern for rule 3 -----> "+result)
            # elif idx==3:
            #     print("pattern for rule 4 -----> "+result)
            
    return pattern_matches

  #----------------------------------------------------
  def predict_setiment(self, sentiment_phrases):
    """ 
      predicts the semantic orientation for each phrase based on average polarity
    """
    polarity_scores = self.calculate_polarity(sentiment_phrases)
    avg_polarity_score = sum(polarity_scores)/len(polarity_scores)
    
    if avg_polarity_score>0:
        return 'pos' 
    else:
        return 'neg'
    
  
  def calculate_polarity(self, sentiment_phrases):
      # Calculates the polarity score of a sentiment phrases 
    h2 = self.negSeedCount + 0.01
    h4 = self.posSeedCount + 0.01

    polarity_scores = []

    for sentiment_phrase in sentiment_phrases:
      words = self.get_words(sentiment_phrase)
      h1 = 0.01
      h3 = 0.01

      if words in self.posHits:
        h1 += self.posHits[words]
      else:
        h1 += 0

      if words in self.negHits:
        h3 += self.negHits[words]
      else:
        h3 += 0

      polarity = math.log2(h1*h2/(h3*h4))
      polarity_scores.append(polarity)
      
    return polarity_scores


  def get_words(self, phrase):
      # Split first 2 words
    phrase = phrase.strip()
    words = phrase.split()

    word1 = words[0].split('_')[0]
    word2 = words[1].split('_')[0]

    return word1 + ' ' + word2

# ------------- add_example and its helper add_example_helper were developed inspired from programming assignment 2    

  def add_example_helper(self, idxs, words_list, seed):
      # Helper function for add Example function
    for i in range(len(idxs)):
        seed_pos = idxs[i]
        start = max(0, seed_pos - self.near)
        end = min(len(words_list), seed_pos + self.near)
        scope = ' '.join(words_list[start:end+1]) 
        scope_list = [scope]
        phrases = self.generate_patterns(scope_list)
        for phrase in phrases:
          words = self.get_words(phrase)
          if seed:
            if words not in self.posHits:
              self.posHits[words] = scope.count(phrase)
            else:
              self.posHits[words] =  self.posHits[words] + scope.count(phrase)
          else:
            if words not in self.negHits:
              self.negHits[words] = scope.count(phrase)
            else:
              self.negHits[words] = self.negHits[words]+scope.count(phrase)

  def add_example(self, klass, words):
    """
    increments self.posSeedCount, and self.negSeedCount based on the words, 
    and compute words freuqnecies that are later used in computing semantic orientation scores.
    """
    
    text = ' '.join(words) 
    words_list = text.split()
    pos_count = text.count(self.posSeed+"_")
    # for pos
    if pos_count != 0 :
      self.posSeedCount += pos_count
      idxs = [ i for i in range(len(words_list)) if self.posSeed in words_list[i] ]
      self.add_example_helper(idxs, words_list, seed=True)
    # for neg
    neg_count = text.count(self.negSeed+"_")
    if neg_count != 0 :
      self.negSeedCount += neg_count
      idxs = [ i for i in range(len(words_list)) if self.negSeed in words_list[i] ]
      self.add_example_helper(idxs,words_list, seed=False)

# ---------------------------- Leveraged from Programming Assignment 2 -----------------------------
  
  def readFile(self, fileName):
      # Leveraged from pa2
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line) 
    f.close()
    result = contents

    return result

  
  def trainSplit(self, trainDir):
      # Leveraged from pa2
      # Split the data the training data
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)

    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
      split.train.append(example)

    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)

    return split

  
  def train(self, split):
      # training function leveraged from pa2
    for example in split.train:
      words = example.words
      self.add_example(example.klass, words)

 
  def crossValidationSplits(self, trainDir):
      # Leveraged from pa2
   # Cross validation helper which splits the data into training and cross-validation based on k-fold
    splits = [] 
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)

    #for fileName in trainFileNames:
    for fold in range(0, self.kFolds):

      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'

        if fileName[2] == str(fold): # split based on the filename number: cv003_11664.txt.out
          split.test.append(example)
        else:
          split.train.append(example)

      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'

        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)

      splits.append(split)

    return splits
  

def test10Fold(args):
    # Leveraged from pa2
  sl = SentimentLexicon()
  splits = sl.crossValidationSplits(args[0])
  avgAccuracy = 0.0
  fold = 0

  for split in splits: 
    classifier = SentimentLexicon()
    accuracy = 0.0

    for example in split.train:
      words = example.words
      classifier.add_example(example.klass, words) 

   
    for example in split.test:
      words = example.words
      sentiment_phrases = classifier.generate_patterns(words) 
      guess = classifier.predict_setiment(sentiment_phrases) 

      if example.klass == guess:
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy

    print('[INFO]\tFold %d Accuracy: %f' % (fold, accuracy)) 
    fold += 1

  avgAccuracy = avgAccuracy / fold
  print('[INFO]\tAccuracy: %f' % avgAccuracy)
    

def classifyDir(trainDir, testDir):
    # Leveraged from pa2
  classifier = SentimentLexicon()
  trainSplit = classifier.trainSplit(trainDir)
  classifier.train(trainSplit) 
  testSplit = classifier.trainSplit(testDir)

  accuracy = 0.0
  for example in testSplit.train:
    words = example.words
    guess = classifier.predict_setiment(words)

    if example.klass == guess:
      accuracy += 1.0
      
  accuracy = accuracy / len(testSplit.train)
  print('[INFO]\tAccuracy: %f' % accuracy)

def main():
    # Leveraged from pa2
  (options, args) = getopt.getopt(sys.argv[1:], '')
  if len(args) == 2:
    classifyDir(args[0], args[1])
  elif len(args) == 1:
    test10Fold(args)

if __name__ == "__main__":
    main()


# run SentimentLexicon.py with arg= ./imdb_tagged/processed_docs/