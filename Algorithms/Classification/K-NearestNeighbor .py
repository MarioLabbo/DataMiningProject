import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Carichiamo i dati
print("Caricamento del dataset...")
df = pd.read_csv("/home/labbo/StudioProjects/PycharmProjects/DataMiningProject/Dataset/dataset_ml_pronto.csv")
X = df.drop(columns=['esi_level'])
y = df['esi_level']

# TRAIN / TEST SPLIT (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Addestramento
print ("Addestramento del KNN in corso...")
knn_clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1) # Guarda i 5 record più simili
knn_clf.fit(X_train, y_train)
print ("Addestramento concluso!")

# Predizione e Valutazione
y_pred_knn = knn_clf.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)

print(f"Accuratezza Globale K-NN: {acc_knn:.4f}")
print("\nReport Dettagliato (K-NN):")
print(classification_report(y_test, y_pred_knn))