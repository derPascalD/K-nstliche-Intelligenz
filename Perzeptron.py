import matplotlib.lines as mlines
import numpy as np
import sklearn.datasets
import h5py
import scipy
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.special import expit
from PIL import Image

"""
-----------------------------------------------------------------------------
Aufgabe 02
-----------------------------------------------------------------------------
"""


def AND(x1, x2):
    w0 = -1.5
    w1 = 1
    w2 = 1
    result = perzeptron(x1, x2, w0, w1, w2)
    return result


def OR(x1, x2):
    w0 = -0.5
    w1 = 1
    w2 = 1
    result = perzeptron(x1, x2, w0, w1, w2)
    return result


def NOT(x):
    w0 = 0.5
    w1 = -1
    result = perzeptron(x, 0, w0, w1, 0)
    return result


def perzeptron(x1, x2, w0, w1, w2):
    sum = w0 + w1 * x1 + w2 * x2
    if sum >= 0:
        return 1
    else:
        return -1


print("AND Gatter")
print(f"AND(0, 0) = {AND(0, 0)}")
print(f"AND(0, 1) = {AND(0, 1)}")
print(f"AND(1, 0) = {AND(1, 0)}")
print(f"AND(1, 1) = {AND(1, 1)}")

print("\nOR Gatter")
print(f"OR(0, 0) = {OR(0, 0)}")
print(f"OR(0, 1) = {OR(0, 1)}")
print(f"OR(1, 0) = {OR(1, 0)}")
print(f"OR(1, 1) = {OR(1, 1)}")

print("\nNOT Gatter")
print(f"NOT(0) = {NOT(0)}")
print(f"NOT(1) = {NOT(1)}")
print("\n")

"""
-----------------------------------------------------------------------------
Aufgabe 03
Datensatz
-----------------------------------------------------------------------------
"""


# Funktion zum Generieren von Punkten
def generate_points(m=100, seed=42):
    X = np.random.uniform(-1, 1, (2, m))  # Erstelle ein Array mit Form (2, m)
    return X


# Funktion zur Generierung einer zufälligen Entscheidungsgrenze
# Rückgabe w: Gewichtungsvektor, der die Entscheidungsgrenze darstellt, der Form (3, 1)
def random_boundary(seed=42):
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
    predictions = np.reshape(predictions, (1, X_ext.shape[1]))  # Forme in (1, m) um
    predictions = np.sign(predictions)  # Vorhersagen auf -1 oder 1 setzen
    return predictions


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


"""
-----------------------------------------------------------------------------
Aufgabe 03
Training
-----------------------------------------------------------------------------
"""


# Definiere die Funktion, die Vorhersagen berechnet
# Eingabe w : Gewichtungsvektor, der das Perzeptron-Modell charakterisiert : der Form (3, 1)
# Eingabe X_ext : Datenmatrix X, erweitert um eine Zeile von Einsen : der Form (3, m)
# Rückgabe predictions : sign(w.transpose * x) : der Form (1, m)


# input w : current weight vector with shape (3,1)
# input x : misclassified data point (should have shape (3,1))
# input y : label of data point x (scalar)

# return new_w : updated weight vector
def weight_update(w, x, y, learning_rate):
    new_w = w + learning_rate * (y - (y * -(1))) * x
    new_w = np.reshape(new_w, (3, 1))
    return new_w


def learn_algo(X, X_ext, learning_rate, num_iterations):
    # Initialize weight vector w_ with 0.
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
        w_ = weight_update(w_, x, Y[0, wrong_point_index], learning_rate)

    return w_, counter_steps


iterations = 1000
total_steps = 0

# Perzeptron Algorithms mehrfach ausführen
X = 0
X_ext = 0
Y = 0
w = 0
w_ = 0

for _ in range(iterations):
    X, X_ext, Y, w = create_dataset(100)
    w_, steps = learn_algo(X, X_ext, 1, 1000)
    total_steps += steps

# Durchschnittliche Anzahl der Schritte berechnen
average_steps = total_steps / iterations
print(f"Durchschnittliche Schritte bis zur Konvergenz bei {iterations} Wiederholung/en: {average_steps}\n")

# Visualisiere die Daten und die Entscheidungsgrenze
fig, ax = plt.subplots()
ax.scatter(X[0, :], X[1, :], marker='o', c=Y, s=25, edgecolor='k')  # flach für Farbgebung

xp = np.array((-1, 1))

plt.title("Alte Grenze- Neue Grenze--")
plt.axis((-1.1, 1.1, -1.1, 1.1))

# Zeichne die neue Grenze (nach dem Training)
yp_new = -(w_[1] / w_[2]) * xp - (w_[0] / w_[2])  # Berechne die y-Werte der Grenze
plt.plot(xp, yp_new, "r--", label="Neue Grenze")

# Zeichne die alte Grenze (vor dem Training)
yp_old = -(w[1] / w[2]) * xp - (w[0] / w[2])
plt.plot(xp, yp_old, "r-", label="Alte Grenze")

plt.show()

"""
-----------------------------------------------------------------------------
Aufgabe 03
Experimente
-----------------------------------------------------------------------------
"""

sum = 0
for i in range(100):
    X, X_ext, Y, w = create_dataset(100)
    w_, steps = learn_algo(X, X_ext, 1, 1000)
    sum += steps

print("m = 100 : Alpha = 1 Average: ", sum / 100)

sum = 0
for i in range(100):
    X, X_ext, Y, w = create_dataset(100)
    w_, steps = learn_algo(X, X_ext, 0.1, 1000)
    sum += steps

print("m = 100 : Alpha = 0.1 Average: ", sum / 100)

sum = 0
for i in range(100):
    X, X_ext, Y, w = create_dataset(1000)
    w_, steps = learn_algo(X, X_ext, 1, 1000)
    sum += steps

print("m = 1000 : Alpha = 1 Average: ", sum / 100)

sum = 0
for i in range(100):
    X, X_ext, Y, w = create_dataset(1000)
    w_, steps = learn_algo(X, X_ext, 0.1, 1000)
    sum += steps

print("m = 1000 : Alpha = 0.1 Average: ", sum / 100)
