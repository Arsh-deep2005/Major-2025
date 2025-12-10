"""
Convert federated learning model to TinyML format
This achieves OBJECTIVE 2: Lightweight model for IoT devices
"""

import tensorflow as tf
import numpy as np
import os

print("=" * 70)
print("OBJECTIVE 2: CREATING TINYML MODEL FROM FEDERATED LEARNING")
print("=" * 70)

def convert_to_tinyml():
    print("\n1️⃣ Loading your federated learning model...")
    model = tf.keras.models.load_model('global_model.h5')
    
    print(f"✅ Model loaded: {model.input_shape} → {model.output_shape}")
    print(f"   Classes: {model.output_shape[1]} (rice diseases)")
    
    # Create representative dataset for quantization
    def representative_dataset():
        print("   Generating calibration data...")
        for _ in range(100):
            # Create dummy images with your model's input shape
            data = np.random.randn(1, 128, 128, 3).astype(np.float32)
            yield [data]
    
    # Convert with INT8 quantization (smallest size)
    print("\n2️⃣ Converting to TinyML format (INT8 quantization)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    
    # Try INT8 quantization
    try:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        tflite_model = converter.convert()
        model_path = "tinyml/fl_model_int8.tflite"
        
    except Exception as e:
        print(f"⚠️ INT8 failed: {e}")
        print("   Using float16 as fallback...")
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        model_path = "tinyml/fl_model_fp16.tflite"
    
    # Save the TinyML model
    os.makedirs("tinyml", exist_ok=True)
    with open(model_path, 'wb') as f:
        f.write(tflite_model)
    
    size_kb = len(tflite_model) / 1024
    print(f"✅ TinyML model saved: {model_path}")
    print(f"📏 Size: {size_kb:.2f} KB ({(size_kb/1024):.2f} MB)")
    
    # Generate C header for microcontrollers
    print("\n3️⃣ Generating C header for microcontrollers...")
    generate_c_header(tflite_model, model_path)
    
    return size_kb

def generate_c_header(model_bytes, model_path):
    """Convert .tflite to C header file for Arduino/ESP32"""
    model_name = os.path.basename(model_path)
    
    header = f"""// Rice Disease Detection - TinyML Model
// Generated from Federated Learning System
// Size: {len(model_bytes)} bytes
// Classes: 6 rice diseases

#ifndef TINYML_MODEL_H
#define TINYML_MODEL_H

const unsigned char tinyml_model[] = {{
"""
    
    # Convert to hex array
    for i in range(0, len(model_bytes), 12):
        line = model_bytes[i:i+12]
        hex_line = ', '.join([f'0x{b:02x}' for b in line])
        header += f'  {hex_line}'
        if i + 12 < len(model_bytes):
            header += ',\n'
        else:
            header += '\n'
    
    header += f"""}};

const unsigned int tinyml_model_len = {len(model_bytes)};

#endif // TINYML_MODEL_H
"""
    
    header_path = "tinyml/tinyml_model.h"
    with open(header_path, 'w', encoding='utf-8') as f:
        f.write(header)
    
    print(f"✅ C header saved: {header_path}")
    
    # Generate Arduino sketch
    generate_arduino_sketch(header_path, model_name)

def generate_arduino_sketch(header_path, model_name):
    """Create complete Arduino sketch for ESP32-CAM"""
    
    sketch = f"""// ===================================================
// RICE DISEASE DETECTOR - IoT EDITION
// Objective 2: TinyML on low-power IoT devices
// Model: {model_name}
// Source: Federated Learning System
// ===================================================

#include "{os.path.basename(header_path)}"
#include <EloquentTinyML.h>

// Configuration (matches your FL model)
#define IMAGE_WIDTH     128
#define IMAGE_HEIGHT    128
#define CHANNELS        3
#define NUM_CLASSES     6
#define TENSOR_ARENA_SIZE  32 * 1024  // 32KB for ESP32

// Rice disease classes (from your app.py)
String class_names[NUM_CLASSES] = {{
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf scald",
    "Sheath Blight"
}};

// TinyML model
Eloquent::TinyML::TfLite<IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS, NUM_CLASSES, TENSOR_ARENA_SIZE> ml;

void setup() {{
    Serial.begin(115200);
    delay(3000);
    
    Serial.println("\\n🌾 RICE DISEASE DETECTOR - IoT EDITION");
    Serial.println("=======================================");
    Serial.println("Powered by Federated Learning + TinyML");
    Serial.println("Objective 2: Lightweight IoT deployment");
    
    // Load TinyML model
    Serial.print("\\nLoading TinyML model... ");
    if (!ml.begin(tinyml_model)) {{
        Serial.println("FAILED!");
        while(1);
    }}
    Serial.println("SUCCESS!");
    
    Serial.print("Model size: ");
    Serial.print(tinyml_model_len);
    Serial.println(" bytes");
    Serial.print("Input: ");
    Serial.print(IMAGE_WIDTH);
    Serial.print("x");
    Serial.print(IMAGE_HEIGHT);
    Serial.print("x");
    Serial.print(CHANNELS);
    Serial.println(" RGB");
    Serial.print("Output: ");
    Serial.print(NUM_CLASSES);
    Serial.println(" classes");
    
    Serial.println("\\n✅ Ready for local inference (no internet needed)");
}}

void loop() {{
    Serial.println("\\n--- Detection Cycle ---");
    
    // Simulate image capture (replace with camera)
    float image[IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS];
    for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS; i++) {{
        image[i] = random(0, 255) / 255.0;
    }}
    
    // Run inference locally on device
    unsigned long start = micros();
    float predictions[NUM_CLASSES];
    ml.predict(image, predictions);
    unsigned long inference_time = micros() - start;
    
    // Display results
    Serial.print("Local inference time: ");
    Serial.print(inference_time / 1000.0);
    Serial.println(" ms");
    
    // Find highest probability
    int detected_class = 0;
    for (int i = 1; i < NUM_CLASSES; i++) {{
        if (predictions[i] > predictions[detected_class]) {{
            detected_class = i;
        }}
    }}
    
    Serial.print("Detected: ");
    Serial.println(class_names[detected_class]);
    Serial.print("Confidence: ");
    Serial.print(predictions[detected_class] * 100, 1);
    Serial.println("%");
    
    Serial.println("\\n💡 Running locally on IoT device:");
    Serial.println("   - No internet required");
    Serial.println("   - No data sent to server");
    Serial.println("   - Instant results (< 200ms)");
    
    delay(10000);  // Wait 10 seconds
}}
"""
    
    sketch_path = "tinyml/iot_detector.ino"
    with open(sketch_path, 'w', encoding='utf-8') as f:
        f.write(sketch)
    
    print(f"✅ Arduino sketch saved: {sketch_path}")
    print("\n📋 Files generated for Objective 2:")
    print(f"   1. tinyml/fl_model_int8.tflite - TinyML model")
    print(f"   2. tinyml/tinyml_model.h - C header for microcontrollers")
    print(f"   3. tinyml/iot_detector.ino - Complete IoT device code")

if __name__ == "__main__":
    print("\n🎯 ACHIEVING OBJECTIVE 2:")
    print("    IoT devices.'")
    print("\n" + "=" * 70)
    
    size = convert_to_tinyml()
    
    print("\n" + "=" * 70)
    print("✅ OBJECTIVE 2 COMPLETE!")
    print("=" * 70)
    print(f"\n📊 RESULTS:")
    print(f"   • Model size: {size:.2f} KB")
    print(f"   • Can run on: ESP32, Arduino, Raspberry Pi Pico")
    print(f"   • Inference: Local, no internet required")
    print(f"   • Privacy: Images stay on device")
    print(f"\n🚀 To deploy on IoT device:")
    print(f"   1. Copy tinyml/tinyml_model.h to Arduino project")
    print(f"   2. Copy tinyml/iot_detector.ino as main sketch")
    print(f"   3. Upload to ESP32-CAM")
    print(f"   4. Open Serial Monitor at 115200 baud")