// Ultra-compact Rice Disease Detector
// Model size: 12.7 KB

#include "micro_model.h"
#include <EloquentTinyML.h>

// Configuration
#define INPUT_SIZE 128*128*3
#define OUTPUT_SIZE 6
#define TENSOR_ARENA_SIZE 8*1024  // 8KB for micro model

Eloquent::TinyML::TfLite<INPUT_SIZE, OUTPUT_SIZE, TENSOR_ARENA_SIZE> ml;

// Disease names
String diseases[OUTPUT_SIZE] = {
  "Bacterial Leaf Blight",
  "Brown Spot",
  "Healthy Rice Leaf",
  "Leaf Blast",
  "Leaf Scald",
  "Sheath Blight"
};

void setup() {
  Serial.begin(115200);
  delay(2000);
  
  Serial.println("Rice Disease Detector - Micro Edition");
  Serial.println("====================================");
  
  Serial.print("Loading 12.7 KB model... ");
  if (ml.begin(micro_model)) {
    Serial.println("OK");
  } else {
    Serial.println("FAILED");
    while(1);
  }
  
  Serial.println("Ready!");
}

void loop() {
  // Create dummy image
  float image[INPUT_SIZE];
  for (int i = 0; i < INPUT_SIZE; i++) {
    image[i] = random(0, 255) / 255.0;
  }
  
  // Run inference
  float predictions[OUTPUT_SIZE];
  unsigned long start = micros();
  ml.predict(image, predictions);
  unsigned long inference_time = micros() - start;
  
  // Show results
  Serial.print("\nInference: ");
  Serial.print(inference_time / 1000.0);
  Serial.println(" ms");
  
  // Find highest probability
  int max_idx = 0;
  for (int i = 1; i < OUTPUT_SIZE; i++) {
    if (predictions[i] > predictions[max_idx]) {
      max_idx = i;
    }
  }
  
  Serial.print("Detected: ");
  Serial.println(diseases[max_idx]);
  Serial.print("Confidence: ");
  Serial.print(predictions[max_idx] * 100, 1);
  Serial.println("%");
  
  delay(5000);
}
