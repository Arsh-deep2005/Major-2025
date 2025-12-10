"""
MINIMAL TinyML conversion script
Converts your trained global_model.h5 to Arduino-compatible format
"""
import tensorflow as tf
import numpy as np
import os

print("=" * 70)
print("MINIMAL TINYML CONVERSION FOR ARDUINO")
print("=" * 70)

# Step 1: Load your trained model
print("\n1️⃣ Loading your trained federated model...")
model = tf.keras.models.load_model('global_model.h5')

print(f"✅ Model loaded successfully!")
print(f"   Input shape: {model.input_shape}")
print(f"   Output classes: {model.output_shape[1]}")

# Step 2: Create tiny directory
os.makedirs("tinyml", exist_ok=True)

# Step 3: Convert to TensorFlow Lite (simple conversion)
print("\n2️⃣ Converting to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Basic optimization

# Convert
tflite_model = converter.convert()

# Save
tflite_path = "tinyml/rice_disease.tflite"
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

size_kb = len(tflite_model) / 1024
print(f"✅ TFLite model saved: {tflite_path}")
print(f"📏 Size: {size_kb:.2f} KB")

# Step 4: Convert to C header for Arduino
print("\n3️⃣ Creating C header file for Arduino...")
with open(tflite_path, 'rb') as f:
    model_bytes = f.read()

# Create C array
c_code = f"""// Rice Disease Detection Model for Arduino
// Generated from Federated Learning
// Model size: {len(model_bytes)} bytes

#ifndef MODEL_H
#define MODEL_H

const unsigned char tinyml_model[] = {{
"""

# Convert bytes to hex (12 per line)
for i in range(0, len(model_bytes), 12):
    chunk = model_bytes[i:i+12]
    hex_values = ', '.join([f'0x{b:02x}' for b in chunk])
    c_code += f'  {hex_values}'
    if i + 12 < len(model_bytes):
        c_code += ',\n'
    else:
        c_code += '\n'

c_code += f"""}};

const unsigned int tinyml_model_len = {len(model_bytes)};

#endif
"""

# Save header
header_path = "tinyml/model.h"
with open(header_path, 'w') as f:
    f.write(c_code)

print(f"✅ C header saved: {header_path}")

# Step 5: Create SIMPLE Arduino sketch
print("\n4️⃣ Creating Arduino sketch...")

arduino_sketch = f"""// Simple Rice Disease Detector for Arduino/ESP32
// Model size: {size_kb:.2f} KB
// 6 rice disease classes

#include "model.h"
#include <EloquentTinyML.h>

// Model configuration
#define INPUT_SIZE 128*128*3  // Your model input size
#define OUTPUT_SIZE 6         // 6 disease classes
#define TENSOR_ARENA_SIZE 32*1024  // 32KB memory

Eloquent::TinyML::TfLite<INPUT_SIZE, OUTPUT_SIZE, TENSOR_ARENA_SIZE> tf;

// Disease names (from your app.py)
String diseases[6] = {{
  "Bacterial Leaf Blight",
  "Brown Spot", 
  "Healthy Rice Leaf",
  "Leaf Blast",
  "Leaf scald",
  "Sheath Blight"
}};

void setup() {{
  Serial.begin(115200);
  delay(3000);
  
  Serial.println("\\n🌾 Rice Disease Detector");
  Serial.println("========================");
  
  Serial.print("Loading model... ");
  if (!tf.begin(tinyml_model)) {{
    Serial.println("FAILED");
    while(1);
  }}
  Serial.println("OK");
  
  Serial.print("Model size: ");
  Serial.print(tinyml_model_len);
  Serial.println(" bytes");
  
  Serial.println("\\nClasses:");
  for (int i = 0; i < 6; i++) {{
    Serial.print("  ");
    Serial.print(i);
    Serial.print(". ");
    Serial.println(diseases[i]);
  }}
  
  Serial.println("\\n✅ Ready!");
}}

void loop() {{
  Serial.println("\\n--- Detection ---");
  
  // Create test input (replace with camera)
  float input[INPUT_SIZE];
  for (int i = 0; i < INPUT_SIZE; i++) {{
    input[i] = random(0, 255) / 255.0;
  }}
  
  // Run inference
  float output[OUTPUT_SIZE];
  unsigned long start = micros();
  tf.predict(input, output);
  unsigned long time = micros() - start;
  
  // Show results
  Serial.print("Time: ");
  Serial.print(time/1000.0);
  Serial.println(" ms");
  
  Serial.println("Probabilities:");
  for (int i = 0; i < 6; i++) {{
    Serial.print("  ");
    Serial.print(diseases[i]);
    Serial.print(": ");
    Serial.print(output[i]*100, 1);
    Serial.println("%");
  }}
  
  // Find highest
  int maxIndex = 0;
  for (int i = 1; i < 6; i++) {{
    if (output[i] > output[maxIndex]) {{
      maxIndex = i;
    }}
  }}
  
  Serial.print("\\nResult: ");
  Serial.println(diseases[maxIndex]);
  Serial.print("Confidence: ");
  Serial.print(output[maxIndex]*100, 1);
  Serial.println("%");
  
  delay(5000);
}}
"""

# Save Arduino sketch
sketch_path = "tinyml/arduino_detector.ino"
with open(sketch_path, 'w') as f:
    f.write(arduino_sketch)

print(f"✅ Arduino sketch saved: {sketch_path}")

print("\n" + "=" * 70)
print("✅ TINYML CONVERSION COMPLETE!")
print("=" * 70)

print("\n📁 Files created in 'tinyml/' folder:")
print("1. rice_disease.tflite - TinyML model")
print("2. model.h - C header for Arduino")
print("3. arduino_detector.ino - Complete Arduino sketch")

print("\n🚀 To upload to Arduino:")
print("1. Install Arduino IDE")
print("2. Install 'EloquentTinyML' library")
print("3. Copy model.h and arduino_detector.ino to same folder")
print("4. Open arduino_detector.ino in Arduino IDE")
print("5. Upload to ESP32/Arduino")
print("6. Open Serial Monitor (115200 baud)")

print("\n🎯 Objective 2 ACHIEVED: Lightweight model for IoT devices!")