# FederatedAveraging
Progetto di Tesi Triennale in Ingegneria Informatica: implementare una simulazione dell'algoritmo Federated Averaging (FedAvg) utilizzando TensorFlow.

L'obiettivo è addestrare un modello di Deep Learning (MLP) su dataset come MNIST o Fashion-MNIST senza che i dati dei singoli utenti vengano mai trasmessi a un server centrale. Solo i pesi del modello vengono condivisi e aggregati.

Il progetto è strutturato in tre varianti:
1. V1 (Base): Simulazione locale su dataset MNIST con partizionamento automatico dei dati.
   
   [main.py](main.py) e [fed_avg.py](fed_avg.py): Versione base per test rapidi

3. V2 (Avanzata): Simulazione su Fashion-MNIST con gestione manuale della distribuzione delle classi (IID/Non-IID) e visualizzazione grafica della distribuzione e dell'accuratezza.
   
   [main_v2.py](main_v2.py) e [fed_avg_v2.py](fed_avg_v2.py): Versione con grafici e controllo della distribuzione dei dati.

5. V3 (Networked): Architettura reale distribuita che utilizza Socket per la comunicazione tra un Server centrale e più processi Client indipendenti (sviluppo solo abbozzato).
   
   [server.py](server.py): Il nodo centrale che gestisce i round, invia i pesi e aggrega i risultati.
   
   [client.py](client.py): Lo script che simula il dispositivo dell'utente, addestra il modello localmente e restituisce i pesi.
