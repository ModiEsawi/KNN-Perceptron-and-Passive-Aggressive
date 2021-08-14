import numpy as np

# shuffling the sets but keeping the same ratio
def unison_shuffled_copies(firstSet, secondSet):
    assert len(firstSet) == len(secondSet)
    p = np.random.permutation(len(firstSet))
    return firstSet[p], secondSet[p]


# used to count number of vectors
def count(givenList):
    newList = []
    for i in givenList:
        if i not in newList:
            newList.append(i)
    return len(newList)


class Perceptron:
    def __init__(self, trainingX, trainingY, learningRate):
        self.learningRate = learningRate
        self.trainingX = trainingX
        self.trainingY = trainingY
        self.numberOfVectors = count(trainingY)
        self.numberOfFeatures = trainingX.shape[1]
        self.weights = np.array(
            [[1, 2, 3, 4, 5, 6, 7, 8, 0, 10, 11, 12, 1], [2, 1, 3, 5, 6, 7, 8, 10, 10, 2, 13, 2, 1],
             [2, 3, 9, 3, 6, 7, 8, 9, 10, 11, 22, 13, 1]], dtype=float)

    # training phase
    def train(self):
        epochs = 10
        for epoch in range(epochs):
            trainX, trainY = unison_shuffled_copies(self.trainingX, self.trainingY)
            for x, y in zip(trainX, trainY):
                y_hat = np.argmax(np.dot(self.weights, x))
                if int(y) != y_hat:
                    self.weights[y_hat, :] = self.weights[y_hat, :] - (self.learningRate * x)
                    self.weights[int(y), :] = self.weights[int(y), :] + (self.learningRate * x)

    # Perceptron answer prediction
    def predict(self, testSet):
        prediction = []
        for sample in testSet:
            prediction.append(np.argmax(np.dot(self.weights, sample)))
        return np.array(prediction)
