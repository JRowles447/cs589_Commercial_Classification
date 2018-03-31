import sys
sys.path.insert(0, 'C:\\Users\\Jason\\Documents\\Umass\\CS589\\HW2\\COMPSCI-589-HW2\\COMPSCI-589-HW2')
from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

from Library.kaggle import kaggleize

class knn():
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k
        self.kneigh = KNeighborsClassifier(n_neighbors = k, weights='distance').fit(X, y)


    # def classify(self):
    #     kneigh = KNeighborsClassifier(n_neighbors = 5, weights='distance')
    #     print(self.X)
    #     print(self.y)
    #     kneigh.fit(self.X, self.y)
    #
    #     return kneigh

    def get_data(FILE, multilabel=None):

        if (multilabel): # for kaggle data
            data = load_svmlight_file(FILE, multilabel=True)
            return data[0].toarray()
        # for training and testing data
        data = load_svmlight_file(FILE)
        return data[0].toarray(), data[1]

if __name__ == '__main__':
    # Examples
    X_train, y_train = knn.get_data('../../Data/HW2.train.txt')
    trainKNN = knn(X_train, y_train, 5)

    X_test, y_test = knn.get_data('../../Data/HW2.test.txt')

    score = trainKNN.kneigh.score(X_train, y_train)
    print("score of training: " + str(score))

    score = trainKNN.kneigh.score(X_test, y_test)

    scorings = []

    print("score of test: " + str(score))
    neighbors = list(range(1, 35))
    # for k in neighbors:
    #     cv_knn = KNeighborsClassifier(n_neighbors=k)
    #     cv_scores = cross_val_score(cv_knn, X_test, y_test, cv=10)
    #     scorings.append(cv_scores.mean())

    # # find the best value of k for hyperparameter
    # k = scorings.index(max(scorings))+ 1
    # print("best k is: " + str(k) +  " with accuracy " + str(scorings[k-1]))
    # trainKNN = knn(X_train, y_train, k)
    # score = trainKNN.kneigh.score(X_test, y_test)
    # print(score)

    trainKNN = knn(X_train, y_train, 32)

    X_kaggle = knn.get_data('../../Data/HW2.kaggle.txt', multilabel=True)
    y_predict = trainKNN.kneigh.predict(X_kaggle)
    zipped = list(zip(X_kaggle, y_predict))
    y_array = np.array(y_predict)
    kaggleize(y_predict, 'kaggle_ready.txt')