import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
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

# Addestramento a tre passi valutandone ognuno
print("Addestramento K-means...")
k_scelto = 3 # Casuale (basarsi sul cluster precedente)

kmeans_step1 = KMeans(n_clusters=k_scelto, init="random", n_init=1, max_iter=1, random_state=42)
kmeans_step2 = KMeans(n_clusters=k_scelto, init="random", n_init=1, max_iter=2, random_state=42)
kmeans_step3 = KMeans(n_clusters=k_scelto, init="random", n_init=1, max_iter=3, random_state=42)

# NOTA: Addestriamo direttamente su X_pca per poter disegnare i confini!
kmeans_step1.fit(X_pca)
kmeans_step2.fit(X_pca)
kmeans_step3.fit(X_pca)
print("Addestramento completato!")

print("\n--- RIDUZIONE DELL'ERRORE (Inerzia) ---")
print("L'inerzia calcola la distanza totale dei pazienti dal loro centroide. Più è bassa, meglio è!")
print(f"Step 1 - Errore iniziale: {kmeans_step1.inertia_:.2f}")
print(f"Step 2 - Errore dopo 1° spostamento: {kmeans_step2.inertia_:.2f}")
print(f"Step 3 - Errore dopo 2° spostamento: {kmeans_step3.inertia_:.2f}")

# 3. Vediamo la distribuzione finale dei pazienti nei 3 cluster
print("\n--- DISTRIBUZIONE FINALE DEI PAZIENTI (Allo Step finale) ---")
valori_unici, conteggi = np.unique(kmeans_step3.labels_, return_counts=True)
for cluster, pazienti in zip(valori_unici, conteggi):
    print(f"Cluster {cluster}: {pazienti} pazienti assegnati")

print("\nProcesso terminato con successo!")