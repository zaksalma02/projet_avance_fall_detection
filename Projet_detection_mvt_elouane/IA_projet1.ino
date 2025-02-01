#include <Arduino.h>
#include "Arduino_BMI270_BMM150.h"

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model.h"

// Seuil pour détecter un mouvement significatif
const float accelerationThreshold = 2.5;
const int numSamples = 119;

int samplesRead = numSamples;

// TensorFlow Lite variables
tflite::AllOpsResolver tflOpsResolver;
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

const char* GESTURES[] = {"punch", "flex"};
#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

// Définir les broches des LEDs
#define LED_R_PIN 22 // Rouge
#define LED_G_PIN 23 // Vert

void setup() {
  // Initialiser les LEDs
  pinMode(LED_R_PIN, OUTPUT);
  pinMode(LED_G_PIN, OUTPUT);

  // Éteindre les LEDs au départ
  digitalWrite(LED_R_PIN, HIGH);
  digitalWrite(LED_G_PIN, HIGH);

  // Initialiser l'IMU
  if (!IMU.begin()) {
    // Si l'IMU ne s'initialise pas, allumer les deux LEDs en guise d'erreur
    digitalWrite(LED_R_PIN, LOW);
    digitalWrite(LED_G_PIN, LOW);
    while (1);
  }

  // Charger le modèle
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    // Si le modèle est incompatible, allumer la LED rouge en guise d'erreur
    digitalWrite(LED_R_PIN, LOW);
    digitalWrite(LED_G_PIN, HIGH);
    while (1);
  }

  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize);
  tflInterpreter->AllocateTensors();
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

void loop() {
  float aX, aY, aZ, gX, gY, gZ;

  // Attendre un mouvement significatif
  while (samplesRead == numSamples) {
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);
      if (aSum >= accelerationThreshold) {
        samplesRead = 0;
        break;
      }
    }
  }

  // Collecter les données et exécuter le modèle
  while (samplesRead < numSamples) {
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      tflInputTensor->data.f[samplesRead * 6 + 0] = (aX + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 1] = (aY + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 2] = (aZ + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 3] = (gX + 2000.0) / 4000.0;
      tflInputTensor->data.f[samplesRead * 6 + 4] = (gY + 2000.0) / 4000.0;
      tflInputTensor->data.f[samplesRead * 6 + 5] = (gZ + 2000.0) / 4000.0;

      samplesRead++;

      if (samplesRead == numSamples) {
        TfLiteStatus invokeStatus = tflInterpreter->Invoke();
        if (invokeStatus != kTfLiteOk) {
          // En cas d'erreur, allumer la LED rouge
          digitalWrite(LED_R_PIN, LOW);
          digitalWrite(LED_G_PIN, HIGH);
          while (1);
        }

        // Lecture des probabilités
        float punchScore = tflOutputTensor->data.f[0];
        float flexScore = tflOutputTensor->data.f[1];

        Serial.print("Le punch score est: ");
        Serial.println(punchScore);

        Serial.print("Le flex score est: ");
        Serial.println(flexScore);

        // Allumer la LED correspondante
        if (punchScore > flexScore) {
          digitalWrite(LED_R_PIN, HIGH); // Éteindre rouge
          digitalWrite(LED_G_PIN, LOW);  // Allumer vert
        } else {
          digitalWrite(LED_R_PIN, LOW);  // Allumer rouge
          digitalWrite(LED_G_PIN, HIGH); // Éteindre vert
        }
      }
    }
  }
}
