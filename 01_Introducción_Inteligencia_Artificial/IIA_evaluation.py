import numpy as np

class BaseMetric:
    def __init__(self, truth, prediction):
        self.truth = truth.reshape(-1,1)
        self.prediction = prediction.reshape(-1,1)

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
    def __call__(self):
        n = self.truth.shape[0]
        return np.sum((self.truth - self.prediction) ** 2) / n


class Var(BaseMetric):
    def __call__(self):
        return np.var((self.truth - self.prediction) ** 2)


class Bias(BaseMetric):
    def __call__(self):
        bias = np.mean((self.truth - self.prediction.mean()))**2
        return bias


def k_folds_model(X_train, y_train, model, error=MSE, k=5):
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
    error = error()

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

    error = np.mean(error_list)

    return error



## llamar todas las métrics =>[cls(y_test.reshape(-1,1), y_predict)() for cls in BaseMetric.__subclasses__() if cls != MSE]

Axes = {0:'X', 1:'Y', 2:'Z'}

for i,j in [(0,1), (1,2), (0,2)]:
    fig0, ax = plt.subplots(figsize=(10,10))
    ax.plot(pred_Gauss[:,i],pred_Gauss[:,j],color='blue',label='Prediction',zorder=3,lw=2)
    ax.plot(pos_gauss[:,i], pos_gauss[:,j], color='grey',ls='--',alpha=0.5,label='Measurement',zorder=1)
    ax.plot(pos[:,i], pos[:,j],color='green',ls='-',alpha=0.5,label='Real',lw=1,zorder=2)
    ax.legend()

    axes = [Axes.get(Ax) for Ax in [i, j]]
    ax.set_title('Comparison between {} predicted, measured and real trajectories'.format(axes))

    plt.show()

# LINEAR REGRESSION MODELS!!

evaluation = MSE()
model = model()
order = 5
eval_array = np.zeros(order)
models = np.array([])

y_models = []

idx = np.argsort(X_test)

for i in range(1, order):
    X_expand = adapt_data_order(X, order)
    model.fit(X_expand, y_train)
    y_pred = model.predict(X_test)

    eval_array[i] = evaluation(y_test, y_pred)

    models = np.append([models, i])
    y_models = np.append([y_models, y_pred], axis=0)

    plt.plot(X_test[idx], y_pred[idx], label=f'Model Order {i}, Linear regression')

plt.legend()
plt.scatter(X_test[idx], y_test[idx], label=f'Model Order {i}, Linear regression')
plt.title('Linear Regression!')


# LOGISTIC REGRESSION MODEL

colors = np.apply_along_axis(lambda x: 'red' if x == 1 else 'blue',1 ,y_test)

a=30
b=100

w = w_mini[:,0]

sns.lineplot(x=[a,b], y=[-a*w[0]/w[1]-w[2]/w[1], -b*w[0]/w[1]-w[2]/w[1]])
sns.scatterplot(x=X_test[:,1], y=X_test[:,0], hue=colors)

"""
¿porqué no me delimita correctamente el límite de decisión?
"""

plt.ylabel('y')
plt.xlabel('X')

plt.title('Logistic Regression Model')
plt.tight_layout()
plt.show()