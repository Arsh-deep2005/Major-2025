import time
import random

def simulate_rice_detector():
    class_names = [
        "Bacterial Leaf Blight",
        "Brown Spot",
        "Healthy Rice Leaf",
        "Leaf Blast",
        "Leaf scald",
        "Sheath Blight"
    ]
    
    recommendations = [
        "Apply streptomycin sulfate (500ppm)",
        "Spray copper oxychloride (0.3%)",
        "No treatment required",
        "Apply tricyclazole (0.1%)",
        "Spray mancozeb (0.2%)",
        "Apply validamycin (0.3%)"
    ]
    
    severities = ["High", "Medium", "None", "High", "Medium", "High"]
    
    print("\n" + "="*41)
    print("   RICE DISEASE DETECTOR")
    print("   (Federated Learning + TinyML)")
    print("="*41)
    
    print("\nInitializing AI model... SUCCESS!")
    print("Model size: 4340 bytes")
    print("\nDetecting 6 rice diseases...")
    
    for i, name in enumerate(class_names):
        print(f"  {i}. {name}")
    
    print("\nReady for detection!")
    print("="*41 + "\n")
    
    while True:
        print("\n" + "="*41)
        print("        RICE LEAF ANALYSIS")
        print("="*41)
        
        print("Capturing rice leaf image...")
        time.sleep(1)
        
        print("Analyzing for diseases...")
        time.sleep(0.5)
        
        # Simulate inference
        inference_time = random.randint(80000, 150000)
        print(f"\nAnalysis time: {inference_time/1000:.1f} ms")
        
        print("\nDISEASE DIAGNOSIS REPORT")
        print("-"*24)
        
        # Generate probabilities
        probs = [random.random() for _ in range(6)]
        total = sum(probs)
        probs = [p/total for p in probs]
        
        for i in range(6):
            print(f"{class_names[i]}: {probs[i]*100:.1f}%")
        
        # Find highest
        max_idx = probs.index(max(probs))
        max_prob = probs[max_idx]
        
        print("\nDIAGNOSIS:")
        print("-"*10)
        print(f"Detected: {class_names[max_idx]}")
        print(f"Confidence: {max_prob*100:.1f}%")
        print(f"Severity: {severities[max_idx]}")
        
        print(f"\nRECOMMENDATION: {recommendations[max_idx]}")
        
        print("\nADVICE:")
        if max_idx == 2:  # Healthy
            print("- Continue regular monitoring")
            print("- Maintain proper irrigation")
            print("- Check nitrogen levels")
        else:
            print("- Isolate affected plants")
            print("- Remove severely infected leaves")
            print("- Apply recommended treatment")
            print("- Avoid overhead watering")
        
        print("\nNext analysis in 10 seconds...")
        print("="*41 + "\n")
        
        time.sleep(10)

if __name__ == "__main__":
    simulate_rice_detector()