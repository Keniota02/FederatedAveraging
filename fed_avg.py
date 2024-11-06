import tensorflow as tf
import tensorflow_datasets as tfds
import random


#Carichiamo il dataset specificato nel main, partizioniamo il train set per ogni client e restituiamo il test set
def load_data(dataset_name, B, num_clients):
    """
    Carica il dataset specificato e lo prepara per l'addestramento.
    """
    (ds_train, ds_test), ds_info = tfds.load(
        dataset_name, 
        split=['train', 'test'], 
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )

    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255., label

    # Prepara i dati
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache().shuffle(ds_info.splits['train'].num_examples).batch(B).prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(B).cache().prefetch(tf.data.AUTOTUNE)

    # Partizionamento del dataset per N client
    client_data = []
    for i in range(num_clients):
        client_data.append(ds_train.shard(num_clients, i))

    return client_data, ds_test


# Funzione per l'aggiornamento locale del client
def client_update(initial_weights, ds_train, E):
    # Crea il modello e applica l'aggiornamento sui dati del client
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.set_weights(initial_weights)
    model.compile(optimizer=tf.keras.optimizers.SGD(0.01),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    
    # Addestra il modello
    model.fit(ds_train, epochs=E)
    
    return model.get_weights()


def start_simulation(K, rounds, C, B, E, dataset_name):
    """
    Funzione che avvia la simulazione di Federated Learning.
    Riceve come parametri K (numero di client), rounds (numero di round),
    C (percentuale di client selezionati), B (dimensione del minibatch) ed E (epoche).
    """
    # Carica e partiziona i dati e ottieni il test_set
    client_data, ds_test = load_data(dataset_name, B, K)

    client_ids = list(range(K))

    # Definisci modello di base
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    # Ottieni pesi iniziali
    w = model.get_weights()

    # Compila il modello prima di eseguire la valutazione
    model.compile(optimizer=tf.keras.optimizers.SGD(0.01),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    #Fase di evaluation del modello globale del server prima del primo round
    print("Valutando il modello prima del primo round...")
    loss, accuracy = model.evaluate(ds_test)
    print(f"Modello iniziale - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    for t in range(rounds):
        print(f"Round {t + 1} iniziato")

        m = max(int(C * K), 1)
        St = random.sample(client_ids, m)

        pesi_cliente = []
        for i in St:
            print(f"\nChiamando il Client({i + 1})...")

            # Passa i pesi al client per l'aggiornamento
            updated_weights = client_update(w, client_data[i], E)  # Aggiornamento dei pesi

            # Raccogli i pesi aggiornati dal client
            pesi_cliente.append(updated_weights)

        # Calcola i nuovi pesi mediando quelli ricevuti dai client
        w = [sum(client_weights[layer] for client_weights in pesi_cliente) / len(pesi_cliente) for layer in range(len(w))]
        print(f"Round {t + 1} completato, pesi aggiornati calcolati.")

        #Fase di valutazione del modello alla fine di ogni round
        print(f"Valutando il modello alla fine del round {t+1}...")
        #Aggiornamento dei pesi nel modello globale
        model.set_weights(w)

        loss, accuracy = model.evaluate(ds_test)
        print(f"Round {t+1} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
