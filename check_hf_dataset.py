from datasets import load_dataset
import os

def check_dataset():
    print("Loading dataset...")
    try:
        ds = load_dataset('TheNetherWatcher/DisasterClassification')
        print(ds)
        print("Features:", ds['train'].features)
        print("Dataset loaded successfully.")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    check_dataset()
