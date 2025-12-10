import tensorflow as tf
import numpy as np
import os

print("=" * 70)
print("FINAL TINYML CONVERSION (AGGRESSIVE QUANTIZATION)")
print("=" * 70)

# Load model
model = tf.keras.models.load_model('global_model.h5')

print(f"Input: {model.input_shape}, Output: {model.output_shape[1]} classes")

# AGGRESSIVE quantization for Arduino
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Use representative dataset
def representative_dataset():
    for _ in range(10):
        data = np.random.randn(1, *model.input_shape[1:]).astype(np.float32)
        yield [data]

converter.representative_dataset = representative_dataset

# Force INT8 quantization (smallest possible)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert
tflite_model = converter.convert()

# Save
os.makedirs("tinyml", exist_ok=True)
with open('tinyml/final_model.tflite', 'wb') as f:
    f.write(tflite_model)

size_kb = len(tflite_model) / 1024
print(f"✅ Final model: {size_kb:.2f} KB")

# Generate minimal Arduino code
arduino_code = """// Ultra-compact Rice Disease Detector
#include <EloquentTinyML.h>

#define NUM_INPUTS 128*128*3
#define NUM_OUTPUTS 6
#define TENSOR_SIZE 16*1024

Eloquent::TinyML::TfLite<NUM_INPUTS, NUM_OUTPUTS, TENSOR_SIZE> tf;

void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("TinyML Ready");
}

void loop() {
    float input[NUM_INPUTS];
    for(int i=0; i<NUM_INPUTS; i++) input[i] = random(0,255)/255.0;
    
    float output[NUM_OUTPUTS];
    tf.predict(input, output);
    
    Serial.print("Inference done: ");
    for(int i=0; i<3; i++) Serial.print(output[i], 2);  // Show first 3
    Serial.println();
    delay(3000);
}
"""

with open('tinyml/minimal.ino', 'w') as f:
    f.write(arduino_code)

print("✅ Created minimal.ino (test sketch)")
print(f"\n🎯 Model ready for Arduino! Size: {size_kb:.2f} KB")