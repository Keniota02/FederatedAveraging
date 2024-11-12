import tensorflow as tf
import tensorflow_datasets as tfds
import random
import matplotlib.pyplot as plt
import numpy as np

def divide_set_uniform(num_clients, images, labels, B):
    num_classes = 10
    images_per_class = len(images) // num_classes  # 6000 immagini per classe
    print(images_per_class)
    classes_per_client = 5  # Ogni client riceve immagini da 5 classi
    images_per_client_per_class = images_per_class // classes_per_client  # Ogni client riceve 1200 immagini per classe

    # Creiamo una lista di immagini suddivise per classe
    class_data = {i: [] for i in range(num_classes)}  # 10 classi di fashion mnist

    # Suddividiamo le immagini per classe
    for img, lbl in zip(images, labels):
        class_data[lbl].append((img, lbl))

    # Creiamo i dati per i client
    client_data = []
    for i in range(num_clients):
        client_images = []
        client_labels = []

        # Ogni client riceve immagini da 5 classi contigue
        for cl in range(classes_per_client):
            # La classe di partenza varia a seconda del client
            start_class = (i + cl) % num_classes
            client_class_images = class_data[start_class][:images_per_client_per_class]

            client_images.extend([item[0] for item in client_class_images])
            client_labels.extend([item[1] for item in client_class_images])

        # Creiamo un dataset da assegnare al client
        client_dataset = tf.data.Dataset.from_tensor_slices((client_images, client_labels))
        client_dataset = client_dataset.shuffle(len(client_images)).batch(B).prefetch(tf.data.AUTOTUNE)
        client_data.append(client_dataset)

    return client_data



def show_client_class_distribution(client_data, num_classes=10):
    """
    Crea un grafico a barre impilate con i client sull'asse x e il numero di campioni per classe sull'asse y.
    Ogni barra è suddivisa in base alla distribuzione delle classi.
    """
    # Inizializza una lista per memorizzare la distribuzione delle classi per ogni client
    class_distribution_per_client = np.zeros((len(client_data), num_classes))  # [client x class]

    # Calcola il numero di campioni per ciascuna classe per ogni client
    for i, client_dataset in enumerate(client_data):
        for batch in client_dataset:
            images, labels = batch
            labels = labels.numpy()  # Converti in numpy array per lavorare sui singoli valori
            for label in labels:
                class_distribution_per_client[i, label] += 1  # Incrementa il conteggio della classe per il client
    
    # Crea un grafico a barre impilate
    plt.figure(figsize=(10, 6))

    # Crea una lista di colori per le classi (per differenziarle visivamente)
    colors = plt.cm.get_cmap("tab10", num_classes)

    # Plot a barre impilate
    bottom = np.zeros(len(client_data))  # Questo terrà traccia di dove inizia ogni segmento di classe

    for class_id in range(num_classes):
        # Estrae il numero di campioni per la classe corrente
        class_counts = class_distribution_per_client[:, class_id]

        # Disegna una barra impilata per questa classe
        plt.bar(range(len(client_data)), class_counts, bottom=bottom, label=f"Classe {class_id}", color=colors(class_id))
        
        # Aggiunge l'altezza di questa classe alla variabile bottom (per impilare correttamente)
        bottom += class_counts

    # Impostazioni del grafico
    plt.title("Distribuzione delle classi per client")
    plt.xlabel("Client")
    plt.ylabel("Numero di Campioni")
    plt.xticks(range(len(client_data)), [f"Client {i}" for i in range(len(client_data))])
    plt.legend(title="Classi")
    plt.grid(axis='y')

    # Mostra il grafico
    plt.show()


#Carichiamo il dataset specificato nel main, partizioniamo il train set per ogni client e restituiamo il test set
def load_data(B, num_clients):
    """
    Carica il dataset specificato e lo prepara per l'addestramento.
    """
    (x_train, y_train),(x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    
    # Normalizza le immagini
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Partizionamento del dataset per N client, in base alle label
    client_data = divide_set_uniform(num_clients, x_train, y_train, B)
    show_client_class_distribution(client_data)

    # Preparazione del test set
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    ds_test = ds_test.batch(B).cache().prefetch(tf.data.AUTOTUNE)

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


def start_simulation(K, rounds, C, B, E):
    """
    Funzione che avvia la simulazione di Federated Learning.
    Riceve come parametri K (numero di client), rounds (numero di round),
    C (percentuale di client selezionati), B (dimensione del minibatch) ed E (epoche).
    """
    # Carica e partiziona i dati e ottieni il test_set
    client_data, ds_test = load_data(B, K)

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

     # Inizializza le liste per registrare l'accuratezza e i round
    accuracy_per_round = [accuracy]
    rounds_list = [0]

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

        # Aggiungi l'accuratezza e il round alla lista per il grafico
        accuracy_per_round.append(accuracy)
        rounds_list.append(t + 1)

    # Visualizza il grafico dell'accuratezza
    plt.figure(figsize=(10, 6))
    plt.plot(rounds_list, accuracy_per_round, marker='o', color='b')
    plt.title("Federated Learning - 5 classi per client")
    plt.xlabel("Round")
    plt.ylabel("Test Accuracy")
    plt.grid()
    plt.show()