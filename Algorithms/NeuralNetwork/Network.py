from sklearn.metrics import accuracy_score, classification_report
from torch import nn, optim
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

# Carichiamo i dati
print("Caricamento del dataset...")
df = pd.read_csv("/home/labbo/StudioProjects/PycharmProjects/DataMiningProject/Dataset/dataset_ml_pronto.csv")
X = df.drop(columns=['esi_level'])
y = df['esi_level']

# TRAIN / TEST SPLIT (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Conversione dei dati in PyTorch Tensors...")

# 1. Convertiamo X_train e X_test in tensori
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

# 2. Convertiamo le Y (Sottraendo 1 per farle partire da 0 invece che da 1 rispettando le condizioni)
y_train_tensor = torch.tensor(y_train.values - 1, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values - 1, dtype=torch.long)

# 3. Creiamo il 'trainloader' (pacchetti di 64 pazienti alla volta)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Input Features: {X_train_tensor.shape[1]}")
print(f"Classi Output: 5 (Mappate da 0 a 4)")

print("Creazione in corso...")
input_size = X_train_tensor.shape[1] # Controlla quante colonne ha il Dataset
output_size = 5   # I 5 livelli di Triage

# Usiamo l'approccio Sequential
model = nn.Sequential(
    nn.Linear(input_size, 64),  # Primo strato nascosto
    nn.ReLU(),                  # Attivazione
    nn.Linear(64, 32),  # Secondo strato nascosto
    nn.ReLU(),                  # Attivazione
    nn.Linear(32, output_size), # Uscita
    nn.LogSoftmax(dim=1)        # Trasforma in Probabilità
)
print("Creazione completata.")

# Presa dal Notebook
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 50  # Le reti neurali hanno bisogno di più 'giri' degli altri algoritmi!

print("Addestramento Rete Neurale in corso...")

# Training loop
for e in range(epochs):
    running_loss = 0

    for patients, labels in trainloader:
        optimizer.zero_grad()  # Pulizia memoria

        output = model(patients)  # FORWARD PASS
        loss = criterion(output, labels)  # CALCOLO ERRORE
        loss.backward()  # BACKWARD PASS (Gradients)
        optimizer.step()  # AGGIORNAMENTO PESI

        running_loss += loss.item()

    # Stampiamo l'errore ogni 10 epoche per non riempire lo schermo
    if (e + 1) % 10 == 0:
        print(f"Epoca {e + 1}/{epochs} - Errore (Loss): {running_loss / len(trainloader):.4f}")

print("Addestramento completato!")

# Spegniamo il calcolo dei gradienti (non stiamo più imparando)
with torch.no_grad():
    # Passiamo tutti i pazienti del test-set nella rete
    logps = model(X_test_tensor)

    # Prendiamo la classe con la probabilità più alta usando torch.exp
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(1, dim=1)

    # Riportiamo le predizioni e i veri valori nel formato Pandas per stamparli
    # Aggiungiamo +1 per tornare alla scala ESI (1-5)
    y_pred_nn = top_class.numpy().squeeze() + 1
    y_test_numpy = y_test_tensor.numpy() + 1

print(" REPORT RETE NEURALE (PyTorch)")
acc_nn = accuracy_score(y_test_numpy, y_pred_nn)
print(f"Accuratezza Globale Rete Neurale: {acc_nn:.4f}\n")
print(classification_report(y_test_numpy, y_pred_nn, zero_division=0))