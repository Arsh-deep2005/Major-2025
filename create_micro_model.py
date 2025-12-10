import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

print("=" * 70)
print("CREATING MICRO MODEL FOR ARDUINO (UNDER 100KB)")
print("=" * 70)

# Get your model's input shape
original_model = tf.keras.models.load_model('global_model.h5')
input_shape = original_model.input_shape[1:]  # (128, 128, 3)
num_classes = original_model.output_shape[1]  # 6

print(f"Original model: {input_shape} → {num_classes} classes")
print(f"Original parameters: {original_model.count_params():,}")

# Create MICRO model (target: < 100KB when quantized)
def create_micro_model():
    model = models.Sequential([
        # Input
        layers.Input(shape=input_shape),
        
        # Tiny conv block 1
        layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Tiny conv block 2
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Tiny conv block 3
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Global pooling (saves HUGE space vs Flatten+Dense)
        layers.GlobalAveragePooling2D(),
        
        # Tiny dense layer
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.3),
        
        # Output
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Create and compile
micro_model = create_micro_model()
micro_model.compile(optimizer='adam', loss='categorical_crossentropy')

print("\n📊 MICRO MODEL ARCHITECTURE:")
micro_model.summary()

print(f"\n📦 SIZE COMPARISON:")
print(f"Original model: {original_model.count_params():,} parameters")
print(f"Micro model:    {micro_model.count_params():,} parameters")
print(f"Reduction:      {((original_model.count_params() - micro_model.count_params()) / original_model.count_params() * 100):.1f}%")

# Train micro model using KNOWLEDGE DISTILLATION
print("\n🚀 Training micro model (knowledge distillation)...")

# Generate synthetic data
print("Creating training data...")
num_samples = 500
X_synthetic = np.random.randn(num_samples, *input_shape).astype(np.float32)

# Get predictions from original (teacher) model
print("Getting teacher predictions...")
y_teacher = original_model.predict(X_synthetic, verbose=1, batch_size=32)

# Train micro model on teacher predictions
print("Training micro model...")
history = micro_model.fit(
    X_synthetic, y_teacher,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Save the trained micro model
micro_model.save('micro_model_trained.h5')
print("✅ Micro model trained and saved")

# Convert to TINY TFLite with aggressive quantization
print("\n🎯 Converting to ultra-small TFLite...")

converter = tf.lite.TFLiteConverter.from_keras_model(micro_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset for quantization
def representative_dataset():
    for i in range(10):
        data = np.random.randn(1, *input_shape).astype(np.float32)
        yield [data]

converter.representative_dataset = representative_dataset

# Force INT8 quantization (smallest possible)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# Save
os.makedirs("tinyml", exist_ok=True)
model_path = "tinyml/micro_model_int8.tflite"
with open(model_path, 'wb') as f:
    f.write(tflite_model)

size_kb = len(tflite_model) / 1024
print(f"✅ Micro model saved: {size_kb:.2f} KB")

if size_kb < 100:
    print("🎉 PERFECT for Arduino!")
elif size_kb < 250:
    print("✅ Good for ESP32")
else:
    print("⚠️  Still a bit large, but workable")

# Generate ARDUINO-READY C header
print("\n🔧 Generating Arduino C header...")

with open(model_path, 'rb') as f:
    model_bytes = f.read()

# Create minimal C header
c_code = f"""// Micro Rice Disease Model for Arduino
// Size: {len(model_bytes)} bytes ({size_kb:.1f} KB)
// Classes: {num_classes}

#ifndef MICRO_MODEL_H
#define MICRO_MODEL_H

const unsigned char micro_model[] = {{
"""

# Add bytes (compact format)
for i in range(0, len(model_bytes), 16):
    chunk = model_bytes[i:i+16]
    hex_line = ', '.join([f'0x{b:02x}' for b in chunk])
    c_code += f'  {hex_line}'
    if i + 16 < len(model_bytes):
        c_code += ',\n'
    else:
        c_code += '\n'

c_code += f"""}};

const unsigned int micro_model_len = {len(model_bytes)};

#endif
"""

header_path = "tinyml/micro_model.h"
with open(header_path, 'w') as f:
    f.write(c_code)

print(f"✅ C header saved: {header_path}")

# Create ULTRA-SIMPLE Arduino sketch
print("\n📝 Creating Arduino sketch...")

sketch = f"""// Ultra-compact Rice Disease Detector
// Model size: {size_kb:.1f} KB

#include "micro_model.h"
#include <EloquentTinyML.h>

// Configuration
#define INPUT_SIZE {input_shape[0]}*{input_shape[1]}*{input_shape[2]}
#define OUTPUT_SIZE {num_classes}
#define TENSOR_ARENA_SIZE 8*1024  // 8KB for micro model

Eloquent::TinyML::TfLite<INPUT_SIZE, OUTPUT_SIZE, TENSOR_ARENA_SIZE> ml;

// Disease names
String diseases[OUTPUT_SIZE] = {{
  "Bacterial Leaf Blight",
  "Brown Spot",
  "Healthy Rice Leaf",
  "Leaf Blast",
  "Leaf Scald",
  "Sheath Blight"
}};

void setup() {{
  Serial.begin(115200);
  delay(2000);
  
  Serial.println("Rice Disease Detector - Micro Edition");
  Serial.println("====================================");
  
  Serial.print("Loading {size_kb:.1f} KB model... ");
  if (ml.begin(micro_model)) {{
    Serial.println("OK");
  }} else {{
    Serial.println("FAILED");
    while(1);
  }}
  
  Serial.println("Ready!");
}}

void loop() {{
  // Create dummy image
  float image[INPUT_SIZE];
  for (int i = 0; i < INPUT_SIZE; i++) {{
    image[i] = random(0, 255) / 255.0;
  }}
  
  // Run inference
  float predictions[OUTPUT_SIZE];
  unsigned long start = micros();
  ml.predict(image, predictions);
  unsigned long inference_time = micros() - start;
  
  // Show results
  Serial.print("\\nInference: ");
  Serial.print(inference_time / 1000.0);
  Serial.println(" ms");
  
  // Find highest probability
  int max_idx = 0;
  for (int i = 1; i < OUTPUT_SIZE; i++) {{
    if (predictions[i] > predictions[max_idx]) {{
      max_idx = i;
    }}
  }}
  
  Serial.print("Detected: ");
  Serial.println(diseases[max_idx]);
  Serial.print("Confidence: ");
  Serial.print(predictions[max_idx] * 100, 1);
  Serial.println("%");
  
  delay(5000);
}}
"""

sketch_path = "tinyml/micro_detector.ino"
with open(sketch_path, 'w') as f:
    f.write(sketch)

print(f"✅ Arduino sketch saved: {sketch_path}")

print("\n" + "=" * 70)
print("✅ MICRO MODEL CREATION COMPLETE!")
print("=" * 70)

print(f"\n📁 Files created:")
print(f"1. micro_model_int8.tflite - {size_kb:.2f} KB model")
print(f"2. micro_model.h - C header file")
print(f"3. micro_detector.ino - Arduino sketch")

print("\n🚀 To upload to Arduino:")
print("1. Copy 'micro_model.h' and 'micro_detector.ino' to Arduino folder")
print("2. Install EloquentTinyML library in Arduino IDE")
print("3. Select board: AI Thinker ESP32-CAM")
print("4. Upload and open Serial Monitor (115200 baud)")

if size_kb < 100:
    print("\n🎉 PERFECT! Your model is now small enough for Arduino!")