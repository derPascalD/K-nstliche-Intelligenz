import numpy as np
from matplotlib import pyplot as plt

"""
-----------------------------------------------------------------------------
Aufgabe 02
Datensatz
-----------------------------------------------------------------------------
"""


# Funktion zum Generieren von Punkten
def generate_points(m=100):
    X = np.random.uniform(-1, 1, (2, m))  # Erstelle ein Array mit Form (2, m)
    return X


# Funktion zur Generierung einer zufälligen Entscheidungsgrenze
# Rückgabe w: Gewichtungsfaktor, der die Entscheidungsgrenze darstellt, der Form (3, 1)
def random_boundary():
    # Generiere zwei zufällige Punkte A und B
    point1 = np.random.uniform(-1, 1, 2)
    point2 = np.random.uniform(-1, 1, 2)

    # Normalvektor
    w1 = -point2[1] + point1[1]  # Differenz der y-Koordinaten
    w2 = point2[0] - point1[0]  # Differenz der x-Koordinaten
    w0 = -(w1 * point1[0] + w2 * point1[1])

    w = np.array([w0, w1, w2])  # Erstelle den Gewichtungsvektor
    w = np.reshape(w, (3, 1))  # Reshape zu (3, 1)

    return w


def predict(w, X_ext):
    # Berechne die Vorhersagen
    predictions = np.matmul(w.T, X_ext)  # w.T hat die Form (1, 3)
    z = np.reshape(predictions, (1, X_ext.shape[1]))  # Forme in (1, m) um
    z = sigmoid(z)
    return z


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def create_dataset(m):
    # Generiere m zufällige Datenpunkte. X hat die Form (2, m)
    X = generate_points(m)

    # Füge eine Zeile von Einsen hinzu. X_ext hat die Form (3, m)
    X_ext = np.vstack((np.ones((1, X.shape[1])), X))

    # Generiere eine zufällige Entscheidungsgrenze. w hat die Form (3, 1)
    w = random_boundary()

    # Generiere Labels. Y hat die Form (1, m)
    Y = predict(w, X_ext)

    return X, X_ext, Y, w


X, X_ext, Y, w = create_dataset(10)
"""
-----------------------------------------------------------------------------
Aufgabe 02
Training
-----------------------------------------------------------------------------
"""
# FERTIG
def weight_update(old_w, X, learning_rate):
    a = predict(old_w, X)
    new_w = old_w - (learning_rate / X.shape[0]) * np.matmul(X, a.T - Y.T)
    new_w = np.reshape(new_w, (3, 1))
    return new_w

# TODO Muss komplett gemacht werden
def learn_algo(X, X_ext, learning_rate, num_iterations):
    # Ini weight vector w_ with 0.
    w_ = np.array([0, 0, 0])
    w_ = np.reshape(w_, (3, 1))
    counter_steps = 0
    for _ in range(num_iterations):
        # calculate predictions for all points
        predictions = predict(w_, X_ext)

        # identify indices of misclassified points
        wrong_points = np.where(Y != predictions)[1]

        # calculate and save number of misclassified points
        wrong_points_counter = len(wrong_points)

        # break if counter_steps = 0
        if wrong_points_counter == 0:
            break
        counter_steps += 1
        # select random misclassified index
        index = np.random.randint(0, len(wrong_points))
        wrong_point_index = wrong_points[index]

        x = np.array([1, X[0, wrong_point_index], X[1, wrong_point_index]])
        x = np.reshape(x, (3, 1))

        # perform one weight update using datapoint at selected index
        w_ = weight_update(w_, X_ext, learning_rate)

    return w_, counter_steps
