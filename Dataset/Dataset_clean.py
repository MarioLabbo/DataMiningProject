import pandas
from sklearn.preprocessing import StandardScaler

df = pandas.read_csv("dataset_progetto_12k.csv")    # Prendiamo il Dataset filtrato

colonne_da_eliminare = ['encounter_id', 'patient_id', 'site_id', 'country',
                        'arrival_timestamp', 'chief_complaint', 'clinical_notes'] # Contenute nel Dataset
df_clean = df.drop(columns=colonne_da_eliminare, errors='ignore') # Eliminazione (ignore serve se non le trova)

df_clean['sex'] = df_clean['sex'].map({'M': 0, 'F': 1}) # Mappatura binaria del sesso (maschio 0 ecc...)

# Eseguiamo il calcolo della Mediana per i valori che potrebbero essere nulli (mancato prelievo del sangue)
# Evitando così valori nulli o errati che andrebbero a causare molti errori
for colonna in df_clean.columns:
    if df_clean[colonna].isnull().sum() > 0:    # Controlla se c'è per colonna almeno 1 valore mancante
        mediana = df_clean[colonna].median()    # Mediana con i valori
        df_clean[colonna] = df_clean[colonna].fillna(mediana)   # Riempe i vuoti con la Mediana

y = df_clean['esi_level']   # Separiamo l'esi_level per evitare "corruzioni"
X = df_clean.drop(columns=['esi_level'])

scaler = StandardScaler()   # Normalizzazoine (Z-Score) per poter calcolare più coerentemente le distanze.
X_scalata = scaler.fit_transform(X) # Valori come età e globuli vengono scalati per poterli usare nei calcoli

df_finale = pandas.DataFrame(X_scalata, columns=X.columns)  # Ricreiamo il Dataset
df_finale['esi_level'] = y.values   # Mettiamoci la colonna target

nome_file_finale = "dataset_ml_pronto.csv"  # Salvataggio finale
df_finale.to_csv(nome_file_finale, index=False)

print(f"\nPulizia terminata! File salvato come '{nome_file_finale}'.")
print(f"Dimensioni finali per l'addestramento: {df_finale.shape}")