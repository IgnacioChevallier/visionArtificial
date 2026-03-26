import csv
import os
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from joblib import load, dump

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def label_to_int(string_label):
    if string_label == 'spade': return 1
    if string_label == 'heart': return 2
    if string_label == 'diamond': return 3
    if string_label == 'club': return 4
    else:
        raise Exception('unknown class_label: ' + string_label)

def int_to_label(int_label):
    if int_label == 1: return 'spade'
    if int_label == 2: return 'heart'
    if int_label == 3: return 'diamond'
    if int_label == 4: return 'club'
    else:
        raise Exception('unknown int_label: ' + str(int_label))

# Agarro las cosas en los archivos las guardo en variables y las mando a train data y labels
def load_training_set():
    train_data = []
    train_labels = []
    with open('generated-files/shapes-hu-moments.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            class_label = row.pop() # saca el ultimo elemento de la lista
            floats = []
            for n in row:
                floats.append(float(n)) # tiene los momentos de Hu transformados a float.
            train_data.append(np.array(floats, dtype=np.float32)) # momentos de Hu
            train_labels.append(np.array([label_to_int(class_label)], dtype=np.int32)) # Resultados
            #Valores y resultados se necesitan por separados
    train_data = np.array(train_data, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int32)
    return train_data, train_labels
# transforma los arrays a arrays de forma numpy

# llama la funcion de arriba, se manda a entrenar y devuelve el modelo entrenado
def train_model():
    train_data, train_labels = load_training_set()
    tree = DecisionTreeClassifier(max_depth=10)
    tree.fit(train_data, train_labels.ravel())
    dump(tree, 'generated-files/model.joblib')  # ← esta línea faltaba
    print("Modelo guardado")
    return tree

def evaluate_model():
    train_data, train_labels = load_training_set()
    model = load('generated-files/model.joblib')

    predictions = model.predict(train_data)
    accuracy = accuracy_score(train_labels.ravel(), predictions)
    print(f"Accuracy sobre el dataset de entrenamiento: {accuracy * 100:.1f}%")

    # Ver predicciones por clase
    for i, (pred, real) in enumerate(zip(predictions, train_labels.ravel())):
        status = "✓" if pred == real else "✗"
        print(f"  {status} real={int_to_label(real)} predicho={int_to_label(pred)}")



train_model()
evaluate_model()
