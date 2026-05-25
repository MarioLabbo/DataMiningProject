import os
import pandas
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()                       # Carica il token
HF_TOKEN = os.getenv("HF_TOKEN")    # Lo prende e usa

if HF_TOKEN is None:                # Controllo di correttezza
    raise ValueError("Token non trovato!")

print("Scaricando il file CSV grezzo in corso, attendere...")

percorso_file_scaricato = hf_hub_download(      # Scarica il Dataset con CSV
    repo_id="olaflaitinen/fedmml-ed-triage",
    filename="fedmml_ed_triage_dataset.csv",
    repo_type="dataset",
    token=HF_TOKEN
)

df_completo = pandas.read_csv(percorso_file_scaricato)  # Leggi il file con Pandas dal percorso temporaneo

print(f"Download completato! Dimensioni: {df_completo.shape}")

df_completo.to_csv("fedmml_ed_triage_dataset.csv", index=False) # Salva il file nella cartella del tuo progetto
print("Dataset salvato localmente con successo!")

sito_scelto = 6 # Sito scelto (12k parametri) per il testing
df_12k = df_completo[df_completo['site_id'] == sito_scelto] # Filtraggio

df_12k.to_csv("dataset_progetto_12k.csv", index=False)
print(f"Dataset finale estratto! Dimensioni: {df_12k.shape}")