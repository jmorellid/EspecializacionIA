import numpy as np

class BaseMetric:
    def __init__(self, truth, prediction):
        self.truth = truth
        self.prediction = prediction

        self.true_positives = np.sum(np.logical_and([truth == 1], [prediction == 1]))
        self.false_positives = np.sum(np.logical_and([truth == 0], [prediction == 1]))
        self.true_negatives = np.sum(np.logical_and([truth == 0], [prediction == 0]))
        self.false_negatives = np.sum(np.logical_and([truth == 1], [prediction == 0]))


class Precision(BaseMetric):
    def __call__(self):
        values_count = len(self.truth)
        Precision = self.true_positives / (self.true_positives + self.false_positives)
        return Precision


class Accuracy(BaseMetric):
    def __call__(self):
        values_count = len(self.truth)
        Accuracy = (self.true_positives + self.true_negatives) / values_count
        return Accuracy


class Recall(BaseMetric):
    def __call__(self):
        values_count = len(self.truth)
        Recall = self.true_positives / (self.true_positives + self.false_negatives)
        return Recall

class F1_score(BaseMetric):
    def __call__(self):
        return self.true_positives / (self.true_positives + 0.5 * (self.false_positives + self.false_negatives))


class MSE(BaseMetric):
    def __call__(self, target, prediction):
        n = target.size
        return np.sum((target - prediction) ** 2) / n


def k_folds_model(X_train, y_train, model, error=MSE(), k=5):
    '''
    Trains <model>, <k> times, each time using 1/<k>*sample as test, and
    the rest as training.
    Each of the <k> times, computes the <error> and keeps the log.

    INPUT
    ---------------------------
    X_train <np.array> Training features
    Y_train <np.array> Array of outcomes
    model <class.BaseModel> Model for training
    error <class.BaseMetric> Error to compute
    k <int> Number of folds

    OUTPUT
    ---------------------------
    mean_error <float> Mean <error> computed over the <k> folds.
    '''


    model = model
    error = error

    chunk_size = int(len(X_train) / k)
    error_list = []
    for i in range(0, len(X_train), chunk_size):
        end = i + chunk_size if i + chunk_size <= len(X_train) else len(X_train)
        new_X_valid = X_train[i: end]
        new_y_valid = y_train[i: end]
        new_X_train = np.concatenate([X_train[: i], X_train[end:]])
        new_y_train = np.concatenate([y_train[: i], y_train[end:]])

        model.fit(new_X_train, new_y_train)
        prediction = model.predict(new_X_valid)
        error_list.append(error(new_y_valid, prediction))

    mean_error = np.mean(error_list)

    return mean_error