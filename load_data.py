from make_data import save_datasets, save_embeddings
import warnings

if __name__ == "__main__":
    print("Loading and saving datasets...")
    save_datasets()
    
    print("Making and saving projections...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        save_embeddings()