import os
import glob
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import argparse
import matplotlib.pyplot as plt

def load_sentences(directory):
    """
    Loads sentences from all text files in the specified directory.
    Each file is assumed to contain one or more lines, where each line is a
    space-separated sequence of tokens.
    
    Args:
        directory (str): Path to the directory containing sentence files.
    
    Returns:
        list[list[str]]: A list of sentences, where each sentence is a list of tokens.
    """
    sentences = []
    # Find all .txt files in the directory.
    files = glob.glob(os.path.join(directory, "*.txt"))
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:  # Only add non-empty lines
                    sentences.append(tokens)
    return sentences

def train_skipgram(sentences, vector_size=100, window=10, min_count=1, epochs=100):
    """
    Trains a skip-gram Word2Vec model on the given sentences.
    
    Args:
        sentences (list[list[str]]): List of tokenized sentences.
        vector_size (int): Dimensionality of the word embeddings.
        window (int): Maximum distance between the current and predicted word.
        min_count (int): Ignores all words with total frequency lower than this.
        epochs (int): Number of iterations (epochs) over the corpus.
        
    Returns:
        gensim.models.Word2Vec: Trained Word2Vec model.
    """
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,           # Use skip-gram; set sg=0 for CBOW.
        workers=4,
        epochs=epochs
    )
    return model

def plot_tsne_embeddings(model, output_filename="embedding_tsne.png"):
    """
    Reduces the word embeddings to 2D using t-SNE and plots them.
    
    Args:
        model (gensim.models.Word2Vec): Trained Word2Vec model.
        output_filename (str): Filename for the saved t-SNE plot image.
    """
    # Retrieve words and their embeddings.
    words = list(model.wv.index_to_key)
    X = model.wv[words]
    
    # Reduce dimensions with t-SNE.
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X)
    
    # Plot the embeddings.
    plt.figure(figsize=(32, 32))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6)
    
    # Annotate each point with the corresponding token.
    for i, word in enumerate(words):
        plt.annotate(word, xy=(X_tsne[i, 0], X_tsne[i, 1]), fontsize=9, alpha=0.8)
    
    plt.title("t-SNE Visualization of Move and Pokémon Embeddings")
    plt.savefig(output_filename)
    plt.show()
    print(f"t-SNE plot saved as {output_filename}")
    print("vocab size:",len(words))

def main():
    # Define the directory containing your space-separated sentence files.
    sentences_dir = "onlineReplayMoveSentences"
    
    # Load all sentences from the files.
    sentences = load_sentences(sentences_dir)
    print(f"Loaded {len(sentences)} sentences from {sentences_dir}.")
    
    # Train a skip-gram model with an increased context window.
    model = train_skipgram(sentences, vector_size=100, window=10, min_count=1, epochs=100)
    print("Skip-gram model trained successfully.")
    
    # Save the trained model to a file.
    model.save("skipgram_model.model")
    print("Model saved as skipgram_model.model.")

    # Create and save a t-SNE plot of the embeddings.
    plot_tsne_embeddings(model, output_filename="embedding.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Word2Vec model and plot embeddings.")
    parser.add_argument("--show", action="store_true", help="Show the t-SNE plot instead of running the main function.")
    args = parser.parse_args()
    
    if args.show:
        model = Word2Vec.load("skipgram_model.model")
        plot_tsne_embeddings(model, output_filename="embedding.png")
    else:
        main()
