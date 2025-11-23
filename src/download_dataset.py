from datasets import load_dataset
from pprint import pprint

def load_data():
    print("Loading dataset from HuggingFace...")
    data = load_dataset("Mels22/SigDetectVerifyFlow")
    print("Dataset structure:")
    pprint(data)
    return data

if __name__ == "__main__":
    load_data()
