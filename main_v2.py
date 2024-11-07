import fed_avg_v2

def main():
    # Parametri personalizzabili per la simulazione
    K = 10       # Numero di client
    rounds = 5   # Numero di round di Federated Learning
    C = 0.3      # Percentuale di client da selezionare in ogni round
    B = 50       # Dimensione del minibatch
    E = 20       # Numero di epoche di addestramento locale per ogni client

    # Avvia la simulazione con i parametri definiti
    fed_avg_v2.start_simulation(K, rounds, C, B, E)

if __name__ == "__main__":
    main()