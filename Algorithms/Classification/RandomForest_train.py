import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Carichiamo i dati
print("Caricamento del dataset...")
df = pd.read_csv("/DataMining/Dataset/dataset_ml_pronto.csv")
X = df.drop(columns=['esi_level'])
y = df['esi_level']

# TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Addestramento
print("Addestramento della Random Forest in corso... (potrebbe richiedere qualche secondo)")
rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
            # L'aggiunta di "class_weight= 'balanced'" serve per penalizzare di più gli errori con pochi record
rnd_clf.fit(X_train, y_train)
print("Addestramento completato!")

# Predizione e Valutazione
y_pred_rf = rnd_clf.predict(X_test)
print("\n--- RISULTATI ---")
print(f"Accuratezza Globale (Accuracy): {accuracy_score(y_test, y_pred_rf):.4f}")
print("\nReport Dettagliato per ogni livello ESI:")
print(classification_report(y_test, y_pred_rf))

# Vediamo quali parametri sono più importanti per decidere il triage
print("\n--- PESO DEGLI ATTRIBUTI ---")
# DataFrame per un output chiaro
weights = pd.DataFrame({
    'Attributo': X_train.columns,
    'Peso': rnd_clf.feature_importances_
}).sort_values(by='Peso', ascending=False)

print(weights.head(10)) # Mostriamo le prime 10 più importanti