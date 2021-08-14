from collections import Counter
import numpy as np


#  calculating the euclidean distance
def euclideanDistance(firstPoint, secondPoint):
    return np.sqrt(np.sum((firstPoint - secondPoint) ** 2))


#  normalize the training sets using the min-max normalization
def normalizeSet(trainingSet, maxValues, minValues):
    newMin = 0
    newMax = 1
    for sample in trainingSet:
        for column in range(trainingSet.shape[1]):
            if newMin != newMax:  # prevent dividing by zero
                normalizedValue = (((sample[column] - minValues[column]) / (maxValues[column] - minValues[column]))
                                   * (newMax - newMin)) + newMin
            else:
                normalizedValue = 1
            sample[column] = normalizedValue


class KNN:

    def __init__(self, trainX, trainY, k):
        self.k = k
        self.trainingX = trainX
        self.trainingY = trainY

    def predict(self, testSet):
        prediction = [self.getYhat(sample) for sample in testSet]
        return np.array(prediction)

    # KNN answer prediction
    def getYhat(self, sample):
        distances = [euclideanDistance(sample, trainingSample) for trainingSample in self.trainingX]
        firstKneighbors = np.argsort(distances)[:self.k]
        neighborsLabels = [self.trainingY[index] for index in firstKneighbors]
        neighborsLabels = sorted(neighborsLabels)
        mostCommon = Counter(neighborsLabels).most_common(1)
        return int(mostCommon[0][0])
