import fed_avg

def main():
    # Parametri personalizzabili per la simulazione
    K = 10       # Numero di client
    rounds = 5   # Numero di round di Federated Learning
    C = 0.3      # Percentuale di client da selezionare in ogni round
    B = 50       # Dimensione del minibatch
    E = 20       # Numero di epoche di addestramento locale per ogni client
    dataset_name = 'mnist'

    # Avvia la simulazione con i parametri definiti
    fed_avg.start_simulation(K, rounds, C, B, E, dataset_name)

if __name__ == "__main__":
    main()