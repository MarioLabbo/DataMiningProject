import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Carichiamo i dati
print("Caricamento del dataset...")
df = pd.read_csv("/home/labbo/StudioProjects/PycharmProjects/DataMiningProject/Dataset/dataset_ml_pronto.csv")
X = df.drop(columns=['esi_level'])
y = df['esi_level']

# TRAIN / TEST SPLIT (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Addestramento
print("Addestramento del Softmax in corso...")
log_reg = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
  # NOTA: max_iter=1000 serve per dare tempo all'algoritmo di trovare i pesi giusti senza dare errori
log_reg.fit(X_train, y_train)
print("Addestramento completato!")

# Valutazione classica
y_pred_log = log_reg.predict(X_test)
acc_log = accuracy_score(y_test, y_pred_log)
print(f"Accuratezza Globale Logistic Regression: {acc_log:.4f}")
print("\nReport Dettagliato:")
print(classification_report(y_test, y_pred_log, zero_division=0))
  # L'ultimo parametro evita "warning" di divisioni per 0. Accade per la variabile ESI=5
  # Il motivo è pressochè lo stesso della RandomForest. Senza specifiche i livelli 4 e 5 sono simili.

# Prendiamo i primi 3 pazienti dal Test Set
pazienti_esempio = X_test.iloc[:3]
veri_codici = y_test.iloc[:3].values

# La funzione predict_proba restituisce le % per ogni classe
probabilita = log_reg.predict_proba(pazienti_esempio)

for i in range(3):

    # Valore reale ESI
    print(f"\nPaziente {i + 1} (Codice Reale Medico: ESI {veri_codici[i]})")
    print("Probabilità calcolate dal Softmax:")

    # Stampiamo le probabilità per le 5 classi formattate in percentuale
    for classe, prob in zip([1, 2, 3, 4, 5], probabilita[i]):
        print(f"  - Probabilità Codice {classe}: {prob * 100:.2f}%")

    # Sceglierà la percentuale più alta
    scelta_modello = log_reg.predict(pazienti_esempio.iloc[[i]])[0]
    print(f"-> Il modello ha assegnato: ESI {scelta_modello}")