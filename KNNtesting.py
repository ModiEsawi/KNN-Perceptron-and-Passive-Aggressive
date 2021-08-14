import sys
import numpy as np
import scipy
from KNN import KNN


class Test_KNN:

    def __init__(self):
        None

    def maxArray(self, arr, k):
        maxVal = arr[0]
        indexMax = 0
        for i in range(1, k):
            if arr[i] > maxVal:
                maxVal = arr[i]
                indexMax = i

        return indexMax

    def distance(self, vectorA, vectorB):
        norm = 0
        for i in range(13):
            currFeature = vectorA[i] - vectorB[i]
            norm = norm + (currFeature * currFeature)
        return np.sqrt(norm)

    def getAccuracy(self, trainX, trainY, newTestSetX, newTestSetY, k):
        knn = KNN(trainX, trainY, k)
        predections = knn.predict(newTestSetX)
        hits = np.sum(newTestSetY == predections)
        calculatedAccuracy = hits / 71

        return calculatedAccuracy

    def knn_fold(self, trainingSetX, trainingSetY, k):

        avg = 0

        # divide the set into 5 parts.
        firstSet = trainingSetX[:71]
        secondSet = trainingSetX[71:142]
        thirdSet = trainingSetX[142:213]
        fourthSet = trainingSetX[213:284]
        fifthSet = trainingSetX[284:355]
        # use first 4 sets as a training set and last set as a validation set.
        newTrainingSetX = np.concatenate((firstSet, secondSet, thirdSet, fourthSet), axis=0)
        newTrainingSetY = trainingSetY[:284]
        newTestY = trainingSetY[284:]
        newTestSet = fifthSet
        avg += self.getAccuracy(newTrainingSetX, newTrainingSetY, newTestSet, newTestY, k)

        # print("first four sets are for training , fifth as a validation set", accuracy)

        # use first first three and fifth sets as a training set and the fourth set as a validation set.
        newTrainingSetX = np.concatenate((firstSet, secondSet, thirdSet, fifthSet), axis=0)
        firstPart = trainingSetY[:213]
        secondPart = trainingSetY[284:355]
        newTrainingSetY = np.concatenate((firstPart, secondPart))
        newTestY = trainingSetY[213:284]
        newTestSet = fourthSet
        avg += self.getAccuracy(newTrainingSetX, newTrainingSetY, newTestSet, newTestY, k)
        # print("first first three and fifth sets as a training set and the fourth set as a validation set", accuracy)

        # use first first two, fourth and fifth sets as a training set and the third set as a validation set.
        newTrainingSetX = np.concatenate((firstSet, secondSet, fourthSet, fifthSet), axis=0)
        firstPart = trainingSetY[:142]
        secondPart = trainingSetY[213:355]
        newTrainingSetY = np.concatenate((firstPart, secondPart))
        newTestY = trainingSetY[142:213]
        newTestSet = thirdSet
        avg += self.getAccuracy(newTrainingSetX, newTrainingSetY, newTestSet, newTestY, k)
        # print("first first two, fourth and fifth sets as a training set and the third set as a validation set", accuracy)

        # use first, third, fourth and fifth sets as a training set and the second set as a validation set.
        newTrainingSetX = np.concatenate((firstSet, thirdSet, fourthSet, fifthSet), axis=0)
        firstPart = trainingSetY[:71]
        secondPart = trainingSetY[142:355]
        newTrainingSetY = np.concatenate((firstPart, secondPart))
        newTestY = trainingSetY[71:142]
        newTestSet = secondSet
        avg += self.getAccuracy(newTrainingSetX, newTrainingSetY, newTestSet, newTestY, k)
        # print("use first, third, fourth and fifth sets as a training set and the second set as a validation set", accuracy)

        # use second, third, fourth and fifth sets as a training set and the first set as a validation set.
        newTrainingSetX = np.concatenate((secondSet, thirdSet, fourthSet, fifthSet), axis=0)
        newTrainingSetY = trainingSetY[71:]
        newTestY = trainingSetY[:71]
        newTestSet = firstSet
        avg += self.getAccuracy(newTrainingSetX, newTrainingSetY, newTestSet, newTestY, k)
        # print("use second, third, fourth and fifth sets as a training set and the first set as a validation set", accuracy)

        return avg / 5

    def shuffle(self, firstSet, secondSet):
        assert len(firstSet) == len(secondSet)
        p = np.random.permutation(len(firstSet))
        return firstSet[p], secondSet[p]

    def accuracy(self, train_x, train_y, learningRate, iterations):

        KNNAccuracy = 0

        for i in range(iterations):
            shuffled_normalized_train_x_array, shuffled_y = self.shuffle(train_x, train_y)
            currAvg = self.knn_fold(shuffled_normalized_train_x_array, shuffled_y, learningRate)
            KNNAccuracy += currAvg

        print(KNNAccuracy / iterations)
