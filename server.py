import socket
import subprocess
import time
import pickle
import tensorflow as tf
import tensorflow_datasets as tfds
import random

# Impostazioni del server
HOST = '127.0.0.1'  # Indirizzo IP locale
PORT = 65432       # Porta su cui il server ascolterà
K = 10               # Numero di client che il server vuole chiamare
rounds = 2
C = 0.2
client_ids = list(range(K))

def receive_data(conn):
    data = b""
    while True:
        part = conn.recv(8192)
        data += part
        if len(part) < 8192:
            break
    return data

def start_server():

    # Definisci modello di base
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    # Ottieni pesi iniziali
    w = model.get_weights()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()
        server_socket.settimeout(10)  # Timeout di 10 secondi
        print("Server in ascolto...")

        for t in range(rounds):
            print(f"Round {t+1} iniziato")

            m = max(int(C*K),1)
            St = random.sample(client_ids, m)

            pesi_cliente = []
            for i in St:
                print(f"\nChiamando il Client({i+1})...")

                # Serializza i pesi
                serialized_w = pickle.dumps(w)

                # Avvia il client come sottoprocesso
                subprocess.Popen(['python', 'client.py'])
                time.sleep(2)  # Attende un momento per dare tempo al client di avviarsi

                # Accetta la connessione da un client
                conn, addr = server_socket.accept()
                with conn:
                    print(f"Connessione stabilita con {addr}")

                    # Invia i pesi iniziali al client
                    conn.sendall(serialized_w)
                    
                    # Ricevi il risultato dal client
                    data = receive_data(conn)
                    if data:
                        peso_agg = pickle.loads(data)
                        print(f"Risultato ricevuto dal Client({i+1}): {peso_agg}")
                        pesi_cliente.append(peso_agg)

                    # Chiudi la connessione con il client prima di passare al successivo
                    print(f"Connessione chiusa con il Client({i+1}).")

            w = [sum(client_weights[layer] for client_weights in pesi_cliente) / K for layer in range(len(w))]
            print(f"Round {t+1} completato, pesi aggiornati calcolati.")


# Avvia il server
start_server()
