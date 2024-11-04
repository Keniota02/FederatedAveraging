import socket
import pickle
import tensorflow as tf
import tensorflow_datasets as tfds

# Impostazioni del client
HOST = '127.0.0.1'  # Indirizzo IP del server (locale)
PORT = 65432       # Porta su cui il server è in ascolto

B = 50 #LOCAL MINIBATCH SIZE
beta = 0 #(split Pk into batches of size B)
E = 20 #number of local epochs

def client_update(initial_weights):
    # Carica il dataset MNIST
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255., label

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache().shuffle(ds_info.splits['train'].num_examples).batch(B).prefetch(tf.data.AUTOTUNE)

    #I test non ci servono
    #ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    #ds_test = ds_test.batch(B).cache().prefetch(tf.data.AUTOTUNE)

    # Definisci il modello e imposta i pesi iniziali
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.set_weights(initial_weights)

    # Compila il modello
    model.compile(
        optimizer=tf.keras.optimizers.SGD(0.01), 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # Addestramento
    model.fit(ds_train, epochs=E)

    return model.get_weights()

# Funzione di ricezione nel client
def receive_data(sock):
    data = b""
    while True:
        part = sock.recv(8192)
        data += part
        if len(part) < 8192:
            break
    return data


def start_client():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((HOST, PORT))

        # Ricevi il parametro serializzato dal server
        data = receive_data(client_socket)
        if data:
            initial_weights = pickle.loads(data)
            print(f"Ricevuto w = {initial_weights} dal server.")

            # Esegui un'operazione su w, ad esempio moltiplica per 2
            updated_weights = client_update(initial_weights)
            
            serialized_weights = pickle.dumps(updated_weights)
            print(f"Inviando {len(serialized_weights)} bytes di pesi aggiornati al server.")
            client_socket.sendall(serialized_weights)

# Avvia il client
start_client()
