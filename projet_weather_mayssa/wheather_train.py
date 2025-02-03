##########
# Telechargé sur google colab
#!pip install wwo-hist
#!pip install pandas==1.3.5
##########




from wwo_hist import retrieve_hist_data # WorldWeatherOnline

frequency="1"
api_key = 'c26b65c5da9f44d4a69104449241312'
location_list = ['canazei']

# retrieve_hist_data returns a list of dataframe(s)
hist_df = retrieve_hist_data(api_key,
                                location_list,
                                '01-JAN-2011',
                                '31-DEC-2020',
                                frequency,
                                location_label = False,
                                export_csv = False,
                                store_df = True)

t_list = hist_df[0].tempC.astype(float).to_list()
h_list = hist_df[0].humidity.astype(float).to_list()
s_list = hist_df[0].totalSnow_cm.astype(float).to_list()

def gen_label(snow):
    if snow > 5:
        return "Yes"
    else:
        return "No"
labels_list = []
for snow, temp in zip(s_list, t_list):
    labels_list.append(gen_label(snow))
    
import pandas as pd

csv_header = ["Temp0", "Temp1", "Temp2", "Humi0", "Humi1", "Humi2", "Snow"]
dataset_df = pd.DataFrame(list(zip(t_list[:-2], t_list[1:-1], t_list[2:], h_list[:-2], h_list[1:-1], h_list[2:], labels_list[2:])), columns=csv_header) 

df0 = dataset_df[dataset_df['Snow'] == "No"]
df1 = dataset_df[dataset_df['Snow'] == "Yes"]
if len(df1.index) < len(df0.index):
    df0_sub = df0.sample(len(df1.index))
    dataset_df = pd.concat([df0_sub, df1])
else:
    df1_sub = df1.sample(len(df0.index))
    dataset_df = pd.concat([df1_sub, df0])   
    
t_list = dataset_df['Temp0'].tolist()
h_list = dataset_df['Humi0'].tolist()
t_list = t_list + dataset_df['Temp2'].tail(2).tolist()
h_list = h_list + dataset_df['Humi2'].tail(2).tolist()

import numpy as np
from numpy import mean
from numpy import std

t_avg = mean(t_list)
h_avg = mean(h_list)
t_std = std(t_list)
h_std = std(h_list)

print("COPY ME!")
print("Température - [MOY, ÉCART TYPE]  ", round(t_avg, 5), round(t_std, 5))
print("Humidité - [MOY, ÉCART TYPE]     ", round(h_avg, 5), round(h_std, 5))



def scaling(val, avg, std):
    return (val - avg) / (std)

dataset_df['Temp0'] = dataset_df['Temp0'].apply(lambda x: scaling(x, t_avg, t_std))
dataset_df['Temp1'] = dataset_df['Temp1'].apply(lambda x: scaling(x, t_avg, t_std))
dataset_df['Temp2'] = dataset_df['Temp2'].apply(lambda x: scaling(x, t_avg, t_std))
dataset_df['Humi0'] = dataset_df['Humi0'].apply(lambda x: scaling(x, h_avg, h_std))
dataset_df['Humi1'] = dataset_df['Humi1'].apply(lambda x: scaling(x, h_avg, h_std))
dataset_df['Humi2'] = dataset_df['Humi2'].apply(lambda x: scaling(x, h_avg, h_std))


import seaborn as sns
import matplotlib.pyplot as plt

t_norm_list = dataset_df['Temp0'].tolist()
h_norm_list = dataset_df['Humi0'].tolist()

# Diagrammes des données brutes
fig, ax = plt.subplots(1, 2)
ax[0].set_title("Température brute")
ax[1].set_title("Humidité brute")
sns.histplot(t_list, ax=ax[0], kde=True)
sns.histplot(h_list, ax=ax[1], kde=True)

# Diagrammes des données normalisées
fig, ax = plt.subplots(1, 2)
sns.histplot(t_norm_list, ax=ax[0], kde=True)
ax[0].set_title("Température normalisée")
ax[1].set_title("Humidité normalisée")
sns.histplot(h_norm_list, ax=ax[1], kde=True)



f_names = dataset_df.columns.values[0:6]  # Noms des colonnes pour les entrées
l_name  = dataset_df.columns.values[6:7] # Nom de la colonne pour la sortie
x = dataset_df[f_names]
y = dataset_df[l_name]



from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
labelencoder.fit(y.Snow)
y_encoded = labelencoder.transform(y.Snow)


from sklearn.model_selection import train_test_split
x_train, x_validate_test, y_train, y_validate_test = train_test_split(
    x, y_encoded, test_size=0.20, random_state=1)
x_test, x_validate, y_test, y_validate = train_test_split(
    x_validate_test, y_validate_test, test_size=0.50, random_state=3)



import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential()
model.add(layers.Dense(12, activation='relu', input_shape=(len(f_names),)))  # Couche cachée
model.add(layers.Dropout(0.2))  # Régularisation
model.add(layers.Dense(1, activation='sigmoid'))  # Couche de sortie
model.summary()  # Affiche le résumé du modèle


model.compile(
    loss='binary_crossentropy',  # Fonction de coût pour une classification binaire
    optimizer='adam',  # Algorithme d'optimisation
    metrics=['accuracy']  # Métrique pour évaluer les performances
)



NUM_EPOCHS = 20
BATCH_SIZE = 64

history = model.fit(
    x_train, y_train,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_validate, y_validate)
)



loss_train = history.history['loss']
loss_val   = history.history['val_loss']
acc_train  = history.history['accuracy']
acc_val    = history.history['val_accuracy']
epochs     = range(1, NUM_EPOCHS + 1)

def plot_train_val_history(x, y_train, y_val, type_txt):
    plt.figure(figsize=(10, 7))
    plt.plot(x, y_train, 'g', label='Training '+type_txt)
    plt.plot(x, y_val, 'b', label='Validation '+type_txt)
    plt.title('Training and Validation '+type_txt)
    plt.xlabel('Epochs')
    plt.ylabel(type_txt)
    plt.legend()
    plt.show()

plot_train_val_history(epochs, loss_train, loss_val, "Loss")
plot_train_val_history(epochs, acc_train, acc_val, "Accuracy")




model.save("/content/snow_forecast.keras")
model = tf.keras.models.load_model("/content/snow_forecast.keras")
model.export("/content/snow_forecast")


y_test_pred = model.predict(x_test)
y_test_pred = (y_test_pred > 0.5).astype("int32")

import sklearn
cm = sklearn.metrics.confusion_matrix(y_test, y_test_pred)


index_names  = ["Actual No Snow", "Actual Snow"]
column_names = ["Predicted No Snow", "Predicted Snow"]

df_cm = pd.DataFrame(cm, index = index_names, columns = column_names)

plt.figure(dpi=150)
sns.heatmap(df_cm, annot=True, fmt='d', cmap="Blues")


TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]

accur  = (TP + TN) / (TP + TN + FN + FP)
precis = TP / (TP + FP)
recall = TP / (TP + FN)
f_score = (2 * recall * precis) / (recall + precis)

print("Accuracy:  ", round(accur, 3))
print("Recall:    ", round(recall, 3))
print("Precision: ", round(precis, 3))
print("F-score:   ", round(f_score, 3))


def representative_data_gen():
    data = tf.data.Dataset.from_tensor_slices(x_test)
    for i_value in data.batch(1).take(100):
        i_value_f32 = tf.dtypes.cast(i_value, tf.float32)
        yield [i_value_f32]


TF_MODEL = "snow_forecast.keras"
converter = tf.lite.TFLiteConverter.from_keras_model(TF_MODEL)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Charger le modèle
model = tf.keras.models.load_model("/content/snow_forecast.keras")

# Voir l'architecture
model.summary()

# Voir les poids et biais
for layer in model.layers:
    print(layer.name, layer.get_weights())


TF_MODEL = tf.keras.models.load_model("/content/snow_forecast.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(TF_MODEL)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Jeu de données représentatif
converter.representative_dataset = tf.lite.RepresentativeDataset(representative_data_gen)

# Optimisation
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Opérateurs supportés
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Types d’entrée/sortie
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

print(type(TF_MODEL))
tflite_model_quant = converter.convert()

tflite_model_quant = converter.convert()

TFL_MODEL_FILE = "snow_model.tflite"
open(TFL_MODEL_FILE, "wb").write(tflite_model_quant)


##########
# Sur google colab
#!apt-get update && apt-get -qq install xxd
#!xxd -i "snow_model.tflite" > model.h
#!sed -i 's/unsigned char/const unsigned char/g' model.h
#!sed -i 's/const/alignas(8) const/g' model.h
############

size_tfl_model = len(tflite_model_quant)
print(len(tflite_model_quant), "bytes")

























    