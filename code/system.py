"""Dummy classification system.

Dummy solution the COM2004/3004 assignment.

REWRITE THE FUNCTIONS BELOW AND REWRITE THIS DOCSTRING

version: v1.0
"""

import math
from typing import List

import numpy as np
from utils import utils
from utils.utils import Puzzle
import scipy.linalg
import scipy.stats
from scipy.stats import multivariate_normal
import random

# The required maximum number of dimensions for the feature vectors.
N_DIMENSIONS = 20

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
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    REWRITE THIS FUNCTION AND THIS DOCSTRING

    Takes the raw feature vectors and reduces them down to the required number of
    dimensions. Note, the `model` dictionary is provided as an argument so that
    you can pass information from the training stage, e.g. if using a dimensionality
    reduction technique that requires training, e.g. PCA.

    The dummy implementation below simply returns the first N_DIMENSIONS columns.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
    if ("PCA" not in model):
        #Computing the principle components
        #Returns 40 principle componenets for each one of the 1560 test images      
        princiComponents = calculate_principal_components(data, 40)    
        pcatrain_data = np.dot((data - np.mean(data)), princiComponents)

        labels = model["labels_train"]

        numberOfIterations = 10

        majorityArray = [0] * 40

        for i in range(0, numberOfIterations):
            randomIndexList, randomLabels = randomPositions(data, labels)

            currentTestData = pcatrain_data[randomIndexList, :]
            currentTrainData = np.delete(pcatrain_data, randomIndexList, axis=0)
            currentTrainLabels = np.delete(np.asarray(model["labels_train"]), randomIndexList, axis=0)


            lettersData = [None] * 26
            #Identifying each individual letter in data.
            for i in range(0, 26):
                lettersData[i] = currentTrainData[currentTrainLabels == chr(ord('A')+ i)]

            bestForwardsPCAs = forwardsChaining(lettersData, currentTestData, randomLabels, None)
            bestBackwardsPCAs = backwardsChaining(lettersData, currentTestData, randomLabels)
            iterationBest = list(set(bestForwardsPCAs).intersection(bestBackwardsPCAs))
            bestPCAs = forwardsChaining(lettersData, currentTestData, randomLabels, iterationBest)
            for x in bestPCAs:
                majorityArray[x] = majorityArray[x] + 1
 
        bestPCAs = []

        for i in range(0, 40):
            if majorityArray[i] >= math.floor(numberOfIterations/2):
                bestPCAs.append(i)

        if len(bestPCAs) < N_DIMENSIONS:
            randomIndexList, randomLabels = randomPositions(data, labels)

            currentTestData = pcatrain_data[randomIndexList, :]
            currentTrainData = np.delete(pcatrain_data, randomIndexList, axis=0)
            currentTrainLabels = np.delete(np.asarray(model["labels_train"]), randomIndexList, axis=0)


            lettersData = [None] * 26
            #Identifying each individual letter in data.
            for i in range(0, 26):
                lettersData[i] = currentTrainData[currentTrainLabels == chr(ord('A')+ i)]

            bestForwardsPCAs = forwardsChaining(lettersData, currentTestData, randomLabels, bestPCAs)

        print(bestPCAs)

        lettersData = [None] * 26
        #Identifying each individual letter in data.
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
        lettersCount[i] = math.ceil(lettersCount[i]/10)

    #Generates a stratified sample of the dataset
    randomlist = [None] * sum(lettersCount)
    randomLabels = [None] * sum(lettersCount)
    count = 0
    while((randomlist[len(randomlist) - 1]) == None):
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
            for x in range(0,40):
                    if x not in bestPCAs:
                        testPCAs = bestPCAs.copy()
                        testPCAs[i] = x
                        score = ratePCAsBinomial(
                            currentTrainData, currentTestData, randomLabels, testPCAs[0:(i+1)]
                        )
                        if(score > iterationHighScore):
                            iterationHighScore = score
                            currentBestNext = testPCAs

            bestPCAs = currentBestNext
    return bestPCAs

def backwardsChaining(lettersData, currentTestData, randomLabels):
    worstPCAs = [None] * 20
    testPCAs = [None] * 20
    iterationHighScore = 0

    for i in range(0,20):
        iterationHighScore = 0
        for x in range(0,40):
                if x not in worstPCAs:
                    testPCAs = list(range(0, 40))
                    testPCAs = [ele for ele in testPCAs if ele not in worstPCAs]
                    testPCAs.remove(x)
                    score = ratePCAsBinomial(
                        lettersData, currentTestData, randomLabels, testPCAs[0:(i+1)]
                    )
                    if(score > iterationHighScore):
                        iterationHighScore = score
                        currentWorstNext = x
        worstPCAs[i] = currentWorstNext

    finalPCAs = list(range(0,40))
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
        covArray[count] = np.cov(newTrain[count], rowvar=0)
    
    #Creates normal dist for training data
    multivarNormalArray = [None] * 26
    for i in range(0,26):
        mean1 = np.mean(newTrain[i], axis=0)
        multivarNormalArray[i] = multivariate_normal(mean=mean1, cov=covArray[i], allow_singular=True)

    #Calcualtes probability of each letter belonging to each class
    probabilityArray = [None] * 26
    for i in range(0,26):
        #print(multivarNormalArray[i].pdf(randomVars[:,:]))
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
    #Calculate 40 then we will select the best 40 later on in the code
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

    print(predictedChars)

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
    return [(0, 0, 1, 1)] * len(words)
