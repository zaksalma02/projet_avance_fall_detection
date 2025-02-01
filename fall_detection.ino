#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
//#include <tensorflow/lite/version.h>

#include "model.h"

// Normalisation: Moyennes et écarts-types calculés lors de l'entraînement (remplacez par vos valeurs réelles)
const float mean[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};  // Remplacez par les moyennes des données d'entraînement
const float std_dev[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};    // Remplacez par les écarts-types des données d'entraînement

const float accelerationThreshold = 2.0; // Seuil d'accélération pour déclencher la détection du mouvement
const int numSamples = 120; // Nombre d'échantillons à collecter avant de faire une prédiction
int samplesRead = 0;
unsigned long lastReadTime = 0;
const unsigned long readInterval = 200; // Intervalle entre les lectures des capteurs

// Variables globales pour TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

constexpr int tensorArenaSize = 24 * 1024; 
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// Fonction sigmoïde pour transformer les résultats en probabilité entre 0 et 1
float sigmoid(float x) {
  return 1.0 / (1.0 + exp(-x));
}

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Initialisation de l'IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // Charger le modèle TensorFlow Lite
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize);
  tflInterpreter->AllocateTensors();
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
  Serial.println("Setup finished");
}

void loop() {
  float aX, aY, aZ, gX, gY, gZ;

  if (millis() - lastReadTime >= readInterval) {
    lastReadTime = millis();  

    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);

      // Vérifier si la somme des accélérations dépasse le seuil d'accélération
      if (aSum >= accelerationThreshold) {
        Serial.println("Motion detected, capturing data...");

        samplesRead = 0; // Réinitialiser les échantillons
        while (samplesRead < numSamples) {
          if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
            IMU.readAcceleration(aX, aY, aZ);
            IMU.readGyroscope(gX, gY, gZ);

            // Normalisation et remplissage des tenseurs pour l'inférence
            aX = (aX - mean[0]) / std_dev[0];
            aY = (aY - mean[1]) / std_dev[1];
            aZ = (aZ - mean[2]) / std_dev[2];

            gX = (gX - mean[3]) / std_dev[3];
            gY = (gY - mean[4]) / std_dev[4];
            gZ = (gZ - mean[5]) / std_dev[5];

            // Remplir les tenseurs d'entrée pour l'inférence
            tflInputTensor->data.f[samplesRead * 6 + 0] = aX;
            tflInputTensor->data.f[samplesRead * 6 + 1] = aY;
            tflInputTensor->data.f[samplesRead * 6 + 2] = aZ;
            tflInputTensor->data.f[samplesRead * 6 + 3] = gX;
            tflInputTensor->data.f[samplesRead * 6 + 4] = gY;
            tflInputTensor->data.f[samplesRead * 6 + 5] = gZ;

            samplesRead++;
          }
        }

        if (samplesRead == numSamples) {
          Serial.println("Running inference..."); 
          TfLiteStatus invokeStatus = tflInterpreter->Invoke();

          if (invokeStatus != kTfLiteOk) {
            Serial.println("Invoke failed!");
            return;
          }

          // Récupération des résultats de l'inférence
          float fallProbability = tflOutputTensor->data.f[1]; // Probabilité de "Fall"

          // Appliquer la fonction sigmoïde pour garantir que la probabilité est entre 0 et 1
          fallProbability = sigmoid(fallProbability);

          // La probabilité de "Normal" est simplement 1 - probabilité de "Fall"
          float normalProbability = 1.0 - fallProbability;

          // Vérifier si la probabilité de "Fall" est supérieure à 0.6 et afficher le résultat
          if (fallProbability > 0.6) {
            Serial.print("Normal Probability: ");
            Serial.println(normalProbability, 6);
            Serial.print("Fall Probability: ");
            Serial.println(fallProbability, 6);
            Serial.println("Classified as Fall");
          } else {
            // Affichage si c'est un mouvement normal
            Serial.print("Normal Probability: ");
            Serial.println(normalProbability, 6);
            Serial.print("Fall Probability: ");
            Serial.println(fallProbability, 6);
            Serial.println("Classified as Normal");
          }
        }
      }
    }
  }
}
