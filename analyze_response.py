import json
import os

TARGET_EXACT = "MJ221982.00 22198 KHAKI PERADA_2025-09-27.jpg"

try:
    with open("response.json", "r") as f:
        data = json.load(f)
        
    results = data.get("results", [])
    print(f"Total Results: {len(results)}")
    
    found_idx = -1
    for i, res in enumerate(results):
        if res["filename"] == TARGET_EXACT:
            found_idx = i
            print(f"✅ Found EXACT MATCH '{res['filename']}' at Rank #{i+1}")
            print(f"   Score: {res['score']:.2f}%")
            print(f"   Motif: {res.get('motif_score', 'N/A')}")
            print(f"   Color: {res.get('color_score', 'N/A')}")
            break
            
    if found_idx == -1:
        print(f"❌ Exact match '{TARGET_EXACT}' NOT found.")
        # Check partials
        print("Checking partial matches...")
        for i, res in enumerate(results):
            if "22198" in res["filename"] and "KHAKI" in res["filename"]:
                print(f"   - Rank #{i+1}: {res['filename']}")
    else:
        print(f"\nConclusion: The item is at position {found_idx+1}. User needs to scroll to row {found_idx // 4 + 1}.")

except Exception as e:
    print(f"Error parsing JSON: {e}")
