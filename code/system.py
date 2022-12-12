"""Dummy classification system.

Dummy solution the COM2004/3004 assignment.

File: System.py
Description: Used by train.py and evaluate.py to learn how to recognise letters i
             a word search and how to solve word searches.
Author: Tom Pearson & Prof Jon Barker
Date Created: 31/10/2022
Date Last Modified: 12/12/2022

version: v1.0
"""

import math
from typing import List

import numpy as np
import numpy.ma as ma

from utils import utils
from utils.utils import Puzzle
import scipy.linalg
import scipy.stats
from scipy.stats import multivariate_normal
import random

# The required maximum number of dimensions for the feature vectors.
N_DIMENSIONS = 20

#Converts an index to its equivalent position in the alphabet
def indexToChar(index) -> chr:
    return chr(ord('A')+index)

def load_puzzle_feature_vectors(image_dir: str, puzzles: List[Puzzle]) -> np.ndarray:
    """Extract raw feature vectors for each puzzle from images in the image_dir.

    OPTIONAL: ONLY REWRITE THIS FUNCTION IF YOU WANT TO REPLACE THE DEFAULT IMPLEMENTATION

    The raw feature vectors are just the pixel values of the images stored
    as vectors row by row. The code does a little bit of work to center the
    image region on the character and crop it to remove some of the background.

    You are free to replace this function with your own implementation but
    the implementation being called from utils.py should work fine. Look at
    the code in utils.py if you are interested to see how it works. Note, this
    will return feature vectors with more than 20 dimensions so you will
    still need to implement a suitable feature reduction method.

    Args:
        image_dir (str): Name of the directory where the puzzle images are stored.
        puzzle (dict): Puzzle metadata providing name and size of each puzzle.

    Returns:
        np.ndarray: The raw data matrix, i.e. rows of feature vectors.

    """
    return utils.load_puzzle_feature_vectors(image_dir, puzzles)

def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """ Takes the raw feature vectors and reduces them down to the required number of
    dimensions using PCA and Normal Distribution.

    Firstly it calculates PCA's up to 30. Then it calculates the best possible PCAs
    to chose out of the 30.

    It calculates this by selecting a stratified sample of letters from the dataset.
        A stratified sample is a sample in which the percentage of each element chosen
        is the same percentage as what occurs in the data set * a multiplier.

    Using the test data the PCAs are then ranked by:
    1) Training data is (data) - (test data) and is seperated into a list where each index
       contains a different array of a particular letter. e.g. [0] = A, [1] = B, [2] = C, etc
    2) Forwards chaining on the training data using test data to calculate score
    3) Completing backwards chaining on the training data using test data to calculate score
    4) Creating a list of PCAs that occured in both
    5) Use forwards chaining to fill in the rest of the list up to 20 elements.
    6) The found PCAs are the saved to a counter
    7) Steps 1-5 are repeated a desired number of times using a new stratified sample
       in each iterations
    
    The top 20 PCAs that occured are saved as the best PCAs. Data is converted to the top 20 PCAs

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """

    #If PCA is not in model then we are in the training phase
    if ("PCA" not in model):
        #Computing the principle components
        #Returns 30 principle componenets for each one of the 1560 test images      
        princiComponents = calculate_principal_components(data, 30)    
        pcatrain_data = np.dot((data - np.mean(data)), princiComponents)

        labels = model["labels_train"]

        NUMBER_OF_ITERATIONS = 50

        majorityArray = [0] * 30 #Contains the times each one of the 30 PCA components has been selected as the best

        for count in range(0, NUMBER_OF_ITERATIONS):
            #Random Sample
            randomIndexList, randomLabels = randomPositions(data, labels)
            #Extract random sample from train data
            currentTestData = pcatrain_data[randomIndexList, :]
            currentTrainData = np.delete(pcatrain_data, randomIndexList, axis=0)
            currentTrainLabels = np.delete(np.asarray(model["labels_train"]), randomIndexList, axis=0)

            #Group each letter up in its own index (A's at [0], B's at [1], etc)
            lettersData = [None] * 26
            for i in range(0, 26):
                lettersData[i] = currentTrainData[currentTrainLabels == chr(ord('A')+ i)]

            #Forwards and backwards chaining to find best PCAs
            bestForwardsPCAs = forwardsChaining(lettersData, currentTestData, randomLabels, None)
            bestBackwardsPCAs = backwardsChaining(lettersData, currentTestData, randomLabels)

            #Groups PCAs that were in both
            iterationBest = list(set(bestForwardsPCAs).intersection(bestBackwardsPCAs))

            #Adds generates PCAs to fill in the gaps
            bestPCAs = forwardsChaining(lettersData, currentTestData, randomLabels, iterationBest)

            #Makes note of the best
            for x in bestPCAs:
                majorityArray[x] = majorityArray[x] + 1

            print("Iteration Number " + str(count) + " Complete!")
 
        bestPCAs = []

        numpyMajority = np.array(majorityArray)

        bestPCAs = np.argpartition(numpyMajority, -20)[-20:]

        bestPCAs = bestPCAs.tolist()

        print(bestPCAs)

        #Identifying each individual letter in data.
        lettersData = [None] * 26
        for i in range(0, 26):
            lettersData[i] = pcatrain_data[np.asarray(model["labels_train"]) == chr(ord('A')+ i)]

        #Converts data to desired feature set
        selectedFeaturesData = [None] * 26
        for i in range(0,26):
            selectedFeaturesData[i] = (lettersData[i])[:, bestPCAs]

        for i in range(0,26):
            model["selectedFeaturesData{}".format(i)] = selectedFeaturesData[i].tolist()

        model["PCA"] = princiComponents.tolist()
        model["train_mean"] = np.mean(data)
        model["best_dimensions"] = bestPCAs

        reduced_data = pcatrain_data[:,bestPCAs]
        print(reduced_data.shape)
    else:
        pcatrain_data = np.dot((data - model["train_mean"]), np.asarray(model["PCA"]))
        reduced_data = pcatrain_data[:, model["best_dimensions"]]

    #Need to subtract the mean of the data set from the training data mean.
    #To the rest of the data including the test data
    return reduced_data

def randomPositions(data, labels):
    lettersCount = [0] * 26

    #Counts the number of each letter
    for x in labels:
        position = ord(x) - 65
        lettersCount[position] = lettersCount[position] + 1


    #Reduces their value to between 1 and 10
    for i in range(0,len(lettersCount)):
        lettersCount[i] = math.ceil(lettersCount[i]/8)
        if lettersCount[i] == 1:
            lettersCount[i] = 4 #Letters that appear the least have minimal impact on the dataset
        elif lettersCount[i] <= 6:
            lettersCount[i] = lettersCount[i] + 3 #Ensures that no letter has few examples


    #Generates a stratified sample of the dataset
    randomlist = [-1] * sum(lettersCount)
    randomLabels = [None] * sum(lettersCount)
    count = 0
    while((randomlist[len(randomlist) - 1]) == -1):
        n = random.randint(0, (len(data) - 1))
        currentLetter = labels[n]
        currentLetterOrd = ord(currentLetter) - 65
        if lettersCount[currentLetterOrd] != 0:
            randomlist[count] = n
            randomLabels[count] = currentLetter
            count = count + 1
    return randomlist, randomLabels

def forwardsChaining(currentTrainData, currentTestData, randomLabels, currentBest):
    if currentBest == None:
        currentBest = []
    bestPCAs = [None] * 20
    testPCAs = [None] * 20
    iterationHighScore = 0
    currentBestNext = 0
    for i in range(0, len(currentBest)):
        bestPCAs[i] = currentBest[i]

    for i in range(len(currentBest),20):
            iterationHighScore = 0
            for x in range(0,30):
                    if x not in bestPCAs:
                        testPCAs = bestPCAs.copy()
                        testPCAs[i] = x
                        score = ratePCAsBinomial(
                            currentTrainData, currentTestData, randomLabels, testPCAs[0:(i+1)]
                        )
                        if(score > iterationHighScore):
                            iterationHighScore = score
                            currentBestNext = testPCAs

            if iterationHighScore == 0:
                testPCAs[i] = 0
                currentBestNext = testPCAs

            bestPCAs = currentBestNext
    return bestPCAs

def backwardsChaining(lettersData, currentTestData, randomLabels):
    worstPCAs = [None] * 20
    testPCAs = [None] * 20
    iterationHighScore = 0

    for i in range(0,20):
        iterationHighScore = 0
        for x in range(0,30):
                if x not in worstPCAs:
                    testPCAs = list(range(0, 30))
                    testPCAs = [ele for ele in testPCAs if ele not in worstPCAs]
                    testPCAs.remove(x)
                    score = ratePCAsBinomial(
                        lettersData, currentTestData, randomLabels, testPCAs[0:(i+1)]
                    )
                    if(score > iterationHighScore):
                        iterationHighScore = score
                        currentWorstNext = x
        worstPCAs[i] = currentWorstNext

    finalPCAs = list(range(0,30))
    finalPCAs = [ele for ele in finalPCAs if ele not in worstPCAs]
    return finalPCAs

def ratePCAsBinomial(train, randomVars, randomVarLabels, features) -> float:
    newTrain = [None] * len(train)
    
    #Converts test data to desired feature set
    for i in range(0,26):
        newTrain[i] = (train[i])[:, features]

    #Converts random vars to desired feature set
    randomVars = randomVars[:, features]

    #Creates cov array for training data
    covArray = [None] * 26
    for count in range(0,26):
        try:
            covArray[count] = np.cov(newTrain[count], rowvar=0)
        except:
            covArray[count] = ma.cov(newTrain[count], rowvar=0)
    
    #Creates normal dist for training data
    multivarNormalArray = [None] * 26
    for i in range(0,26):
        mean1 = np.mean(newTrain[i], axis=0)
        try:
            multivarNormalArray[i] = multivariate_normal(mean=mean1, cov=covArray[i], allow_singular=True)
        except ValueError:
            covArray[i] = ma.cov(newTrain[count], rowvar=0)
            multivarNormalArray[i] = multivariate_normal(mean=mean1, cov=covArray[i], allow_singular=True)


            

    #Calcualtes probability of each letter belonging to each class
    probabilityArray = [None] * 26
    for i in range(0,26):
        probabilityArray[i] = multivarNormalArray[i].pdf(randomVars[:,:])
    p = np.vstack(probabilityArray)
    index = np.argmax(p, axis=0)
    predictedChars = [None] * len(randomVarLabels)
    for i in range(0, len(randomVarLabels)):
        predictedChars[i] = indexToChar(index[i])

    correct = [None] * len(predictedChars)

    for i in range(0, len(predictedChars)):
        if predictedChars[i] == randomVarLabels[i]:
            correct[i] = True
        else:
            correct[i] = False

    return (np.sum(correct) * 100.0 / len(randomVarLabels))

def calculate_principal_components(data: np.ndarray, numOfDimensions: int) -> np.ndarray:
    covx = np.cov(data, rowvar=0)
    N = covx.shape[0] #N is the number of pixes (existing dimentions)

    #w isn't needed as it is an eigenvalue and we only want the vector
    #Calculate 30 then we will select the best 30 later on in the code
    w, v = scipy.linalg.eigh(covx, eigvals=(N - numOfDimensions, N - 1)) 

    v = np.fliplr(v) #Reverses order of elements along axis 1
    return v

def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    REWRITE THIS FUNCTION AND THIS DOCSTRING

    This is your classifier's training stage. You need to learn the model parameters
    from the training vectors and labels that are provided. The parameters of your
    trained model are then stored in the dictionary and returned. Note, the contents
    of the dictionary are up to you, and it can contain any serializable
    data types stored under any keys. This dictionary will be passed to the classifier.

    The dummy implementation stores the labels and the dimensionally reduced training
    vectors. These are what you would need to store if using a non-parametric
    classifier such as a nearest neighbour or k-nearest neighbour classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    # The design of this is entirely up to you.
    # Note, if you are using an instance based approach, e.g. a nearest neighbour,
    # then the model will need to store the dimensionally-reduced training data and labels
    # e.g. Storing training data labels and feature vectors in the model.
    model = {}
    model["labels_train"] = labels_train.tolist()
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()

    return model

def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Dummy implementation of classify squares.

    REWRITE THIS FUNCTION AND THIS DOCSTRING

    This is the classification stage. You are passed a list of unlabelled feature
    vectors and the model parameters learn during the training stage. You need to
    classify each feature vector and return a list of labels.

    In the dummy implementation, the label 'E' is returned for every square.

    Args:
        fvectors_train (np.ndarray): feature vectors that are to be classified, stored as rows.
        model (dict): a dictionary storing all the model parameters needed by your classifier.

    Returns:
        List[str]: A list of classifier labels, i.e. one label per input feature vector.
    """

    selectedFeaturesData = [None] * 26

    for i in range(0,26):
        selectedFeaturesData[i] = np.array(model["selectedFeaturesData{}".format(i)])

    #Creates cov array for training data
    covArray = [None] * 26
    for i in range(0,26):
        covArray[i] = np.cov(selectedFeaturesData[i], rowvar=0)
    
    #Creates normal dist for training data
    multivarNormalArray = [None] * 26
    for i in range(0,26):
        mean1 = np.mean(selectedFeaturesData[i], axis=0)
        multivarNormalArray[i] = multivariate_normal(mean=mean1, cov=covArray[i], allow_singular=True)

    #Calcualtes probability of each letter belonging to each class
    probabilityArray = [None] * 26
    for i in range(0,26):
        probabilityArray[i] = multivarNormalArray[i].pdf(fvectors_test[:,:])
    p = np.vstack(probabilityArray)
    index = np.argmax(p, axis=0)
    predictedChars = [None] * len(fvectors_test[:])
    for i in range(0, len(predictedChars)):
        predictedChars[i] = indexToChar(index[i])

    return predictedChars

def find_words(labels: np.ndarray, words: List[str], model: dict) -> List[tuple]:
    """Dummy implementation of find_words.

    REWRITE THIS FUNCTION AND THIS DOCSTRING

    This function searches for the words in the grid of classified letter labels.
    You are passed the letter labels as a 2-D array and a list of words to search for.
    You need to return a position for each word. The word position should be
    represented as tuples of the form (start_row, start_col, end_row, end_col).

    Note, the model dict that was learnt during training has also been passed to this
    function. Most simple implementations will not need to use this but it is provided
    in case you have ideas that need it.

    In the dummy implementation, the position (0, 0, 1, 1) is returned for every word.

    Args:
        labels (np.ndarray): 2-D array storing the character in each
            square of the wordsearch puzzle.
        words (list[str]): A list of words to find in the wordsearch puzzle.
        model (dict): The model parameters learned during training.

    Returns:
        list[tuple]: A list of four-element tuples indicating the word positions.
    """

    minWordLength = len(min(words, key=len))
    largestWordLength = len(max(words, key=len))
    rangeOfLetters =  largestWordLength - minWordLength + 1
    
    sortedWords = [None] * rangeOfLetters

    count = 0
    for word in words:
        wordLen = len(word)
        wordIndex = wordLen - minWordLength 
        if sortedWords[wordIndex] == None:
            sortedWords[wordIndex] = ([wordClass(word, count)]) 
        else:
            sortedWords[wordIndex].append(wordClass(word, count)) 
        count = count + 1

    maxYCoordinate = len(labels) - 1
    maxXCoordinate = len(labels[0]) - 1

    for currentYCoordinate in range(0,len(labels)):
        for currentXCoordinate in range(0,len(labels[currentYCoordinate])):
            possiblePaths = calculatePossibleWordLength(currentXCoordinate, currentYCoordinate, maxXCoordinate, maxYCoordinate, largestWordLength)
            while np.any(possiblePaths >= minWordLength - 1):
                #North
                if possiblePaths[0] >= minWordLength - 1:
                    iterationXCoordinate = currentXCoordinate
                    iterationYCoordinate = currentYCoordinate - possiblePaths[0]
                    currentLettersInRange = getLettersBetweenRange(currentXCoordinate,
                        currentYCoordinate, (0), (-1), iterationXCoordinate, iterationYCoordinate, labels)

                    for word in sortedWords[possiblePaths[0] - minWordLength + 1]:
                        word.compareWord(currentLettersInRange, (currentYCoordinate, currentXCoordinate, iterationYCoordinate, iterationXCoordinate))

                    possiblePaths[0] = possiblePaths[0] - 1

                    
                #North East
                if possiblePaths[1] >= minWordLength - 1:
                    iterationXCoordinate = currentXCoordinate + possiblePaths[1]
                    iterationYCoordinate = currentYCoordinate - possiblePaths[1]
                    currentLettersInRange = getLettersBetweenRange(currentXCoordinate,
                        currentYCoordinate, (1), (-1), iterationXCoordinate, iterationYCoordinate, labels)

                    for word in sortedWords[possiblePaths[1] - minWordLength + 1]:
                        word.compareWord(currentLettersInRange, (currentYCoordinate, currentXCoordinate, iterationYCoordinate, iterationXCoordinate))


                    possiblePaths[1] = possiblePaths[1] - 1

                #East
                if possiblePaths[2] >= minWordLength - 1:
                    iterationXCoordinate = currentXCoordinate + possiblePaths[2]
                    iterationYCoordinate = currentYCoordinate 
                    currentLettersInRange = getLettersBetweenRange(currentXCoordinate,
                        currentYCoordinate, (1), (0), iterationXCoordinate, iterationYCoordinate, labels)

                    for word in sortedWords[possiblePaths[2] - minWordLength + 1]:
                        word.compareWord(currentLettersInRange, (currentYCoordinate, currentXCoordinate, iterationYCoordinate, iterationXCoordinate))

                    possiblePaths[2] = possiblePaths[2] - 1

                #South East
                if possiblePaths[3] >= minWordLength - 1:
                    iterationXCoordinate = currentXCoordinate + possiblePaths[3]
                    iterationYCoordinate = currentYCoordinate + possiblePaths[3]
                    currentLettersInRange = getLettersBetweenRange(currentXCoordinate,
                        currentYCoordinate, (1), (1), iterationXCoordinate, iterationYCoordinate, labels)

                    for word in sortedWords[possiblePaths[3] - minWordLength + 1]:
                        word.compareWord(currentLettersInRange, (currentYCoordinate, currentXCoordinate, iterationYCoordinate, iterationXCoordinate))

                    possiblePaths[3] = possiblePaths[3] - 1


                #South
                if possiblePaths[4] >= minWordLength - 1:
                    iterationXCoordinate = currentXCoordinate
                    iterationYCoordinate = currentYCoordinate + possiblePaths[4]
                    currentLettersInRange = getLettersBetweenRange(currentXCoordinate,
                        currentYCoordinate, (0), (1), iterationXCoordinate, iterationYCoordinate, labels)

                    for word in sortedWords[possiblePaths[4] - minWordLength + 1]:
                        word.compareWord(currentLettersInRange, (currentYCoordinate, currentXCoordinate, iterationYCoordinate, iterationXCoordinate))

                    possiblePaths[4] = possiblePaths[4] - 1


                #South West
                if possiblePaths[5] >= minWordLength - 1:
                    iterationXCoordinate = currentXCoordinate - possiblePaths[5]
                    iterationYCoordinate = currentYCoordinate + possiblePaths[5]
                    currentLettersInRange = getLettersBetweenRange(currentXCoordinate,
                        currentYCoordinate, (-1), (1), iterationXCoordinate, iterationYCoordinate, labels)

                    for word in sortedWords[possiblePaths[5] - minWordLength + 1]:
                        word.compareWord(currentLettersInRange, (currentYCoordinate, currentXCoordinate, iterationYCoordinate, iterationXCoordinate))

                    possiblePaths[5] = possiblePaths[5] - 1


                #West
                if possiblePaths[6] >= minWordLength - 1:
                    iterationXCoordinate = currentXCoordinate - possiblePaths[6]
                    iterationYCoordinate = currentYCoordinate
                    currentLettersInRange = getLettersBetweenRange(currentXCoordinate,
                        currentYCoordinate, (-1), (0), iterationXCoordinate, iterationYCoordinate, labels)

                    for word in sortedWords[possiblePaths[6] - minWordLength + 1]:
                        word.compareWord(currentLettersInRange, (currentYCoordinate, currentXCoordinate, iterationYCoordinate, iterationXCoordinate))

                    possiblePaths[6] = possiblePaths[6] - 1


                #North West
                if possiblePaths[7] >= minWordLength - 1:
                    iterationXCoordinate = currentXCoordinate - possiblePaths[7]
                    iterationYCoordinate = currentYCoordinate - possiblePaths[7]
                    currentLettersInRange = getLettersBetweenRange(currentXCoordinate,
                        currentYCoordinate, (-1), (-1), iterationXCoordinate, iterationYCoordinate, labels)

                    for word in sortedWords[possiblePaths[7] - minWordLength + 1]:
                        word.compareWord(currentLettersInRange, (currentYCoordinate, currentXCoordinate, iterationYCoordinate, iterationXCoordinate))

                    possiblePaths[7] = possiblePaths[7] - 1




    #Output predicted positions
    output = []
    for word in words:
        currentLenWords = len(word) - minWordLength
        for currentWord in sortedWords[currentLenWords]:
            if word == currentWord.word:
                output.append(currentWord.predictedPosition)


    return(output)

def calculatePossibleWordLength(xCoordinate, yCoordinate, maxX, maxY, maxWordLen) -> np.array:
    northLen = yCoordinate
    eastLen = maxX - xCoordinate
    southLen = maxY - yCoordinate
    westLen = xCoordinate

    if northLen > eastLen:
        northEastLen = eastLen
    else:
        northEastLen = northLen

    if southLen > eastLen:
        southEastLen = eastLen
    else:
        southEastLen = southLen

    if southLen > westLen:
        southWestLen = westLen
    else:
        southWestLen = southLen

    if northLen > westLen:
        northWestLen = westLen
    else:
        northWestLen = northLen

    results = np.array([northLen, northEastLen, eastLen, southEastLen, southLen, southWestLen, westLen, northWestLen])

    tooBig = np.nonzero(results > maxWordLen - 1)

    results[tooBig] = (maxWordLen - 1)

    return results

def getLettersBetweenRange(startX, startY, rulesForX, rulesForY, endX, endY, letters) -> str:
    word = "" + letters[startY][startX]
    
    while (startX != endX or startY != endY):
        startX = startX + rulesForX
        startY = startY + rulesForY
        word += (letters[startY][startX])

    return word



class wordClass:
    def __init__(self, wordName, wordIndex):
        self.word = wordName
        self.wordLength = len(wordName)
        self.predictedPosition = (0,0,0,0)
        self.wordLength = 0
        self.correctLetters = 0
        self.found = False
        self.wordIndex = wordIndex

    def compareWord(self, providedLetters: str, predictedPos):
        if self.found:
            return
        count = 0
        currentCorrect = 0
        for let in providedLetters:
            if self.word[count].lower() == let.lower():
                currentCorrect += 1
            count += 1
        if currentCorrect > self.correctLetters:
            self.correctLetters = currentCorrect
            self.predictedPosition = predictedPos
            if currentCorrect == len(self.word):
                self.found = True
