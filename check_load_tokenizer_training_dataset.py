from datasets import load_from_disk


DATASET_PATH = "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset"


if __name__ == "__main__":
    try:
        dataset = load_from_disk(DATASET_PATH)
        print("LOAD_OK")
        print(type(dataset))
    except Exception as error:
        print("LOAD_FAILED")
        print(type(error).__name__)
        print(str(error))
