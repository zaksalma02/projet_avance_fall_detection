#include <Arduino_HS300x.h>
#include "model.h"
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

#define SENSOR HS300x

const tflite::Model* tflu_model            = nullptr;
tflite::MicroInterpreter* tflu_interpreter = nullptr;
TfLiteTensor* tflu_i_tensor                = nullptr;
TfLiteTensor* tflu_o_tensor                = nullptr;

float tflu_o_scale = 1.0f;
int32_t tflu_o_zero_point = 0;

constexpr int t_sz = 4096; // Taille en octets
uint8_t tensor_arena[t_sz] __attribute__((aligned(16)));

tflite::AllOpsResolver tflu_ops_resolver;


constexpr int num_hours = 3;
int8_t t_vals[num_hours] = {0};
int8_t h_vals[num_hours] = {0};
int cur_idx = 0;

float tflu_i_scale = 1.0f;
int32_t tflu_i_zero_point = 0;

constexpr int num_reads = 3;

void setup() {

  Serial.begin(9600);  
  while (!Serial);  
  if (!SENSOR.begin()) {
      Serial.println("Failed sensor initialization!");   
      while (1); 
     }
   Serial.print("Test Temperature = ");  
   Serial.print(SENSOR.readTemperature(), 2);  
   Serial.println(" °C");  Serial.print("Test Humidity = ");  
   Serial.print(SENSOR.readHumidity(), 2);  
   Serial.println(" %");

   tflu_model = tflite::GetModel(snow_model_tflite);

   static tflite::MicroInterpreter static_interpreter(
    tflu_model,
    tflu_ops_resolver,
    tensor_arena,
    t_sz
    );
    tflu_interpreter = &static_interpreter;
  
  tflu_interpreter->AllocateTensors();
  tflu_i_tensor = tflu_interpreter->input(0);
  tflu_o_tensor = tflu_interpreter->output(0);

  const auto* i_quant = reinterpret_cast<TfLiteAffineQuantization*>(tflu_i_tensor->quantization.params);
  const auto* o_quant = reinterpret_cast<TfLiteAffineQuantization*>(tflu_o_tensor->quantization.params);
  tflu_i_scale = i_quant->scale->data[0];
  tflu_i_zero_point = i_quant->zero_point->data[0];
  tflu_o_scale = o_quant->scale->data[0];
  tflu_o_zero_point = o_quant->zero_point->data[0];



}

void loop() {
    float t = 0.0f;
    float h = 0.0f;
    for (int i = 0; i < num_reads; ++i) {
        t += SENSOR.readTemperature();
        h += SENSOR.readHumidity();
        delay(3000);
    }
    t /= (float)num_reads;
    h /= (float)num_reads;

  // print each of the sensor values
   Serial.print("Temperature = ");
   Serial.print(t);
   Serial.println(" °C");

   Serial.print("Humidité = ");
   Serial.print(h);
   Serial.println(" %");

    constexpr float t_mean = 2.08993f;
    constexpr float h_mean = 87.22773f;
    constexpr float t_std = 6.82158f;
    constexpr float h_std = 14.21543f;

    t = (t - t_mean) / t_std;
    h = (h - h_mean) / h_std;

    t = (t / tflu_i_scale);
    t += (float)tflu_i_zero_point;
    h = (h / tflu_i_scale);
    h += (float)tflu_i_zero_point;

    t_vals[cur_idx] = t;
    h_vals[cur_idx] = h;
    cur_idx = (cur_idx + 1) % num_hours;

    int32_t idx0 = cur_idx;
    int32_t idx1 = (cur_idx - 1 + num_hours) % num_hours;
    int32_t idx2 = (cur_idx - 2 + num_hours) % num_hours;

    tflu_i_tensor->data.int8[0] = t_vals[idx2];
    tflu_i_tensor->data.int8[1] = t_vals[idx1];
    tflu_i_tensor->data.int8[2] = t_vals[idx0];
    tflu_i_tensor->data.int8[3] = h_vals[idx2];
    tflu_i_tensor->data.int8[4] = h_vals[idx1];
    tflu_i_tensor->data.int8[5] = h_vals[idx0];

    tflu_interpreter->Invoke();

    delay(3000);
 
    float out_int8 = tflu_o_tensor->data.int8[0];
    float out_f = (out_int8 - tflu_o_zero_point);
    out_f *= tflu_o_scale;

    if (out_f < 0.5) {
        Serial.println("Probabilite outf=");
        Serial.println(out_f );
        Serial.println("No, it does not snow");
    } else {
        Serial.println("Probabilite outf=");
        Serial.println(out_f );
        Serial.println("Yes, it snows");
    }

}
