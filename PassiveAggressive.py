import numpy as np
from Perceptron import unison_shuffled_copies, count


class PassiveAggressive:
    def __init__(self, trainingX, trainingY):
        self.trainingX = trainingX
        self.trainingY = trainingY
        self.numberOfVectors = count(trainingY)
        self.numberOfFeatures = trainingX.shape[1]
        self.weights = np.array(
            [[1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1],
             [1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1],
             [1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1]],
            dtype=float)

    # training phase
    def train(self):
        epochs = 10
        for epoch in range(epochs):
            trainX, trainY = unison_shuffled_copies(self.trainingX, self.trainingY)
            for x, y in zip(trainX, trainY):
                weightsCopy = self.weights
                withoutY = np.delete(weightsCopy, int(y), 0)
                y_hat = np.argmax(np.dot(withoutY, x))
                if y == 0:  # this means that all original vectors moved to the left
                    y_hat += 1
                elif y == 1:  # only the last vector moved to the left
                    if y_hat == 1:
                        y_hat += 1
                if int(y) != y_hat:
                    tau = (self.loss(x, int(y), y_hat)) / (2 * np.dot(x, x))
                    self.weights[y_hat, :] = self.weights[y_hat, :] - (tau * x)
                    self.weights[int(y), :] = self.weights[int(y), :] + (tau * x)

    # Passive aggressive answer prediction
    def predict(self, testSet):
        prediction = []
        for sample in testSet:
            prediction.append(np.argmax(np.dot(self.weights, sample)))
        return np.array(prediction)

    # calculating the loss function
    def loss(self, x, y, y_hat):
        rightVector = self.weights[y]
        wrongVector = self.weights[y_hat]
        return max(0, 1 - (np.dot(rightVector, x)) + (np.dot(wrongVector, x)))
