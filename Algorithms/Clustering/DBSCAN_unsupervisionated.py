import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Carichiamo i dati
print("Caricamento del dataset...")
df = pd.read_csv("/home/labbo/StudioProjects/PycharmProjects/DataMiningProject/Dataset/dataset_ml_pronto.csv")
X = df.drop(columns=['esi_level'])
y = df['esi_level']

# TRAIN / TEST SPLIT (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Riduzione delle dimensioni
pca = PCA(n_components=2)
# Trasformiamo la nostra X (15 colonne) in X_pca (2 colonne)
X_pca = pca.fit_transform(X)

# Addestramento
print("Addestramento del DBSCAN in corso...")
dbscan = DBSCAN(eps=1.0, min_samples=5)
# NOTA: si può modificare eps e min_samples per vedere come cambia il grafico!
dbscan.fit(X_pca)
print("Addestramento completato!")

# Risultati
print(f"Cluster trovati: {np.unique(dbscan.labels_)}")
print(f"Numero di cluster: {np.unique(dbscan.labels_).shape[0] - (1 if -1 in dbscan.labels_ else 0)}")
print(f"Anomalie: {list(dbscan.labels_).count(-1)}")