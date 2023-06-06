import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import os
import glob
import math

os.chdir('/Users/ludo/Documents/src/AI4Health')

# Liste des noms de fichiers CSV
fichiers_csv = [[file for file in glob.glob("*/*/*/follow_traj.txt")]]

# Liste pour stocker les données de tous les fichiers
donnees = []


filtering_data = ['records/test1/traj/follow_traj.txt',
                  'records/BO2504/test/follow_traj.txt',
                  'records/fuzzy/tt/follow_traj.txt',
                  'records/test1/fx/follow_traj.txt',
                  # Above Problematic data : Below test data not relevant
                  'records/test1/av/follow_traj.txt',
                  'records/test2/traj/follow_traj.txt',
                  'records/test2/av/follow_traj.txt',
                  'records/traj/test/follow_traj.txt',
                  'records/move/follow_traj.txt',
                  'NO_march1-3/follow_traj.txt',
                  'NO_march6-10/follow_traj.txt',
                  'GU_marche1/follow_traj.txt',
                  'GU_marche2/follow_traj.txt'
                  ]

# Charger chaque fichier CSV et ajouter les données à la liste
for fichier_csv in fichiers_csv[0]:
    print(fichier_csv)
    fd_stat = os.stat(fichier_csv)
    if (fd_stat.st_size != 0) and (fichier_csv not in filtering_data):
        data = pd.read_csv(fichier_csv, sep='\s+', header=None)
        donnees.append(data)

# Concaténer toutes les données en un seul DataFrame
donnees_combinees = pd.concat(donnees)

# ## Convertir les caractères '-' en NaN
# donnees_combinees = donnees_combinees.replace('-', np.nan)
# ## Supprimer les champs qui ne sont pas des nombres
# donnees_combinees = donnees_combinees.apply(pd.to_numeric, errors='coerce')
# ## Supprimer les lignes contenant des NaN
# donnees_combinees = donnees_combinees.dropna()

donnees_combinees = donnees_combinees.dropna()

# Diviser les données en features et labels
X = donnees_combinees.iloc[:, 4:9].values
y = donnees_combinees.iloc[:, 18].values

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Normaliser les données
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Normaliser les données de sortie
min_y_train = min(y_train)
range_y_train = max(y_train) - min(y_train)
y_train = (y_train)/range_y_train
y_test = (y_test)/range_y_train


# Créer le modèle du réseau de neurones
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(
    4, activation='sigmoid', input_shape=(X_train.shape[1],)))
# model.add(tf.keras.layers.Dense(4, activation='sigmoid'))
model.add(tf.keras.layers.Dense(4, activation='sigmoid'))
model.add(tf.keras.layers.Dense(4, activation='sigmoid'))
model.add(tf.keras.layers.Dense(4, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

#
epochs = 20

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]


# Compiler le modèle
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(X_train, y_train, epochs=epochs,
                    batch_size=8, callbacks=callbacks, validation_data=(X_test, y_test))


# plot
accuracy = history.history['accuracy']
loss = history.history['loss']
val_accuracy = history.history['val_accuracy']
val_loss = history.history['val_loss']

# Tracer la précision
plt.plot(accuracy, label='Entraînement')
plt.plot(val_accuracy, label='Validation')
plt.title('Précision du modèle')
plt.xlabel('Époque')
plt.ylabel('Précision')
plt.legend()
plt.show()

# Tracer la perte
plt.plot(loss, label='Entraînement')
plt.plot(val_loss, label='Validation')
plt.title('Perte du modèle')
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.legend()
plt.show()
# Tracer la performance du modèle

# Évaluer le modèle
yres = model.predict(X_test)
plt.plot(yres, label='Prédiction')
plt.plot(y_test, label='Réel')
plt.title('Performance du modèle')
plt.xlabel('Échantillon')
plt.ylabel('Valeur')
plt.legend()
plt.show()
