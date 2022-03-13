import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('Wine_Dataset.csv')

print(df.describe())

# Normalization fields
norm_fields = ['fixed acidity', 'residual sugar', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

def normalize(df, fields):
    for field in fields:
        df[field] = (df[field] - df[field].min())/(df[field].max() - df[field].min())
    return df

df = normalize(df, norm_fields)
print(df.describe())

X = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 
        'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'color']].to_numpy()
Y = df['quality'].to_numpy()

# training 80% testing 20%
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20)
kernels = ['rbf', 'linear', 'poly']

# experimentation of different svm kerns with varying C values
def model_experimentation():
    for kernel in kernels:
        reg_param_C = []
        if kernel == 'rbf' or kernel == 'poly':
            reg_param_C = [1, 10, 50, 100]
        else:
            reg_param_C = [1, 10, 20, 30]

        for c in reg_param_C:
            svm = SVC(C=c, kernel=kernel, gamma='auto')
            svm.fit(train_x, train_y)
            mean_acc = svm.score(test_x, test_y)
            print('mean accuracy: ', mean_acc, '\t', 'kernel: ', kernel, '\t', 'C: ', c)

# experimenting best kernal with varying C and Gamma values
def best_kernel_experimentation():
    c_vals = [1, 10, 50, 100]
    gamma_vals = [1, 10, 50, 100]

    for c in c_vals:
        for gamma in gamma_vals:
            svm = SVC(C=c, kernel='rbf', gamma=gamma)
            svm.fit(train_x, train_y)
            mean_acc = svm.score(test_x, test_y)
            print('mean accuracy: {0:.5g}, \t C: {1}, \t gamma: {2}'.format(mean_acc, c, gamma))

model_experimentation()
best_kernel_experimentation()