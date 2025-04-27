import numpy as np
import random
class RNNWordPredictor:
    def __init__(self, embedding_dim=10, hidden_dim=20, learning_rate=0.01, clip_value=5.0):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.clip_value = clip_value
        self.W_embed = None
        self.W_xh = None
        self.W_hh = None
        self.W_hy = None
        self.b_h = None
        self.b_y = None
        self.word_to_ix = {}
        self.ix_to_word = {}
        self.vocab_size = 0
    def initialize_model(self, data):
        print("Initializing model...")
        vocab = sorted(set(word for seq in data for word in seq))
        self.word_to_ix = {w: i for i, w in enumerate(vocab)}
        self.ix_to_word = {i: w for w, i in self.word_to_ix.items()}
        self.vocab_size = len(vocab)
        random.seed(42)
        np.random.seed(42)
        self.W_embed = np.random.randn(self.vocab_size, self.embedding_dim) * np.sqrt(1. / self.vocab_size)
        self.W_xh = np.random.randn(self.embedding_dim, self.hidden_dim) * np.sqrt(1. / self.embedding_dim)
        self.W_hh = np.random.randn(self.hidden_dim, self.hidden_dim) * np.sqrt(1. / self.hidden_dim)
        self.W_hy = np.random.randn(self.hidden_dim, self.vocab_size) * np.sqrt(1. / self.hidden_dim)
        self.b_h = np.zeros((self.hidden_dim,))
        self.b_y = np.zeros((self.vocab_size,))
        print(f"Model initialized with vocabulary size: {self.vocab_size}")
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    def cross_entropy_loss(self, pred, target_idx):
        return -np.log(pred[target_idx] + 1e-8)
    def clip_gradients(self, *grads):
        for g in grads:
            np.clip(g, -self.clip_value, self.clip_value, out=g)
    def train(self, data, target_words, num_epochs=1000, early_stopping_patience=50):
        print_section_header("Training RNN Model")
        min_loss = float('inf')
        patience_counter = 0
        best_weights = None
        for epoch in range(num_epochs):
            total_loss = 0
            for sequence, target_word in zip(data, target_words):
                try:
                    inputs = [self.word_to_ix[w] for w in sequence[:-1]]
                    target_idx = self.word_to_ix[target_word]
                except KeyError:
                    print(f"Warning: Skipping sequence {sequence} due to unknown word.")
                    continue

                embed_vectors = [self.W_embed[ix] for ix in inputs]
                h_prev = np.zeros((self.hidden_dim,))
                hs, xs = [], []
                for x in embed_vectors:
                    xs.append(x)
                    h_linear = np.dot(x, self.W_xh) + np.dot(h_prev, self.W_hh) + self.b_h
                    h = np.tanh(h_linear)
                    hs.append(h)
                    h_prev = h
                y_linear = np.dot(h, self.W_hy) + self.b_y
                y_pred = self.softmax(y_linear)
                loss = self.cross_entropy_loss(y_pred, target_idx)
                total_loss += loss

                dW_xh = np.zeros_like(self.W_xh)
                dW_hh = np.zeros_like(self.W_hh)
                dW_hy = np.zeros_like(self.W_hy)
                db_h = np.zeros_like(self.b_h)
                db_y = np.zeros_like(self.b_y)
                dW_embed = np.zeros_like(self.W_embed)
                dy = y_pred.copy()
                dy[target_idx] -= 1
                dW_hy += np.outer(hs[-1], dy)
                db_y += dy
                dh = np.dot(self.W_hy, dy)

                for t in reversed(range(len(xs))):
                    h = hs[t]
                    dh_raw = dh * (1 - h ** 2)
                    dW_xh += np.outer(xs[t], dh_raw)
                    prev_h = hs[t - 1] if t != 0 else np.zeros_like(h)
                    dW_hh += np.outer(prev_h, dh_raw)
                    db_h += dh_raw
                    dW_embed[inputs[t]] += np.dot(self.W_xh, dh_raw)
                    dh = np.dot(self.W_hh, dh_raw)

                self.clip_gradients(dW_xh, dW_hh, dW_hy, db_h, db_y, dW_embed)
                self.W_embed -= self.learning_rate * dW_embed
                self.W_xh -= self.learning_rate * dW_xh
                self.W_hh -= self.learning_rate * dW_hh
                self.W_hy -= self.learning_rate * dW_hy
                self.b_h -= self.learning_rate * db_h
                self.b_y -= self.learning_rate * db_y

            avg_loss = total_loss / len(data)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{num_epochs}: Average Loss = {avg_loss:.4f}")
                print(f"Progress: {((epoch + 1) / num_epochs * 100):.1f}%")
            if avg_loss < min_loss:
                min_loss = avg_loss
                patience_counter = 0
                best_weights = {
                    'W_embed': self.W_embed.copy(), 'W_xh': self.W_xh.copy(),
                    'W_hh': self.W_hh.copy(), 'W_hy': self.W_hy.copy(),
                    'b_h': self.b_h.copy(), 'b_y': self.b_y.copy()
                }
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    self.W_embed, self.W_xh, self.W_hh = best_weights['W_embed'], best_weights['W_xh'], best_weights['W_hh']
                    self.W_hy, self.b_h, self.b_y = best_weights['W_hy'], best_weights['b_h'], best_weights['b_y']
                    break
    def predict_next_word(self, sequence_input, top_k=1):
        try:
            inputs = [self.word_to_ix[w] for w in sequence_input]
        except KeyError:
            return None, None
        h_prev = np.zeros((self.hidden_dim,))
        for x_idx in inputs:
            x = self.W_embed[x_idx]
            h_linear = np.dot(x, self.W_xh) + np.dot(h_prev, self.W_hh) + self.b_h
            h_prev = np.tanh(h_linear)
        y_linear = np.dot(h_prev, self.W_hy) + self.b_y
        y_pred = self.softmax(y_linear)
        if top_k == 1:
            pred_idx = np.argmax(y_pred)
            return self.ix_to_word[pred_idx], y_pred[pred_idx]
        else:
            top_indices = np.argsort(y_pred)[-top_k:][::-1]
            return [(self.ix_to_word[i], y_pred[i]) for i in top_indices], None
    def save_predictions(self, test_sequences, filename="predictions.txt"):
        print(f"\nSaving predictions to {filename}...")
        with open(filename, 'w') as f:
            f.write("RNN Word Prediction Results\n")
            f.write("=" * 30 + "\n\n")
            for seq in test_sequences:
                predictions, _ = self.predict_next_word(seq, top_k=3)
                f.write(f"Input: {' '.join(seq)}\n")
                if predictions:
                    f.write("Top 3 Predictions:\n")
                    for word, prob in predictions:
                        f.write(f"  - {word}: {prob:.4f}\n")
                else:
                    f.write("Error: Unknown word in sequence\n")
                f.write("\n")
        print(f"Predictions saved to {filename}")
def print_section_header(title):
    print(f"\n{'=' * 50}")
    print(f"{title.center(50)}")
    print(f"{'=' * 50}\n")
def print_table(headers, rows):
    col_widths = [max(len(str(item)) for item in col) for col in zip(headers, *rows)]
    header_row = "│ " + " │ ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " │"
    separator = "├" + "─".join("─" * (w + 2) for w in col_widths) + "┤"
    top_border = "┬".join("─" * (w + 2) for w in col_widths)
    top_border = "┌" + top_border[1:-1] + "┐"
    bottom_border = "┴".join("─" * (w + 2) for w in col_widths)
    bottom_border = "└" + bottom_border[1:-1] + "┘"
    print(top_border)
    print(header_row)
    print(separator)
    for row in rows:
        row_str = "│ " + " │ ".join(str(item).ljust(w) for item, w in zip(row, col_widths)) + " │"
        print(row_str)
    print(bottom_border)
def main():
    print(f"{' RNN Word Predictor '.center(50, '*')}\n")
    data = [
        ["The", "dog", "runs", "fast"],
        ["A", "cat", "jumps", "high"],
        ["The", "bird", "flies", "away"],
        ["A", "fish", "swims", "deep"]
    ]
    target_words = ["fast", "high", "away", "deep"]
    test_sequences = [
        ["The", "dog", "runs"],
        ["A", "cat", "jumps"],
        ["The", "bird", "flies"],
        ["A", "fish", "swims"],
        ["The", "cat", "runs"]
    ]
    model = RNNWordPredictor(embedding_dim=10, hidden_dim=20, learning_rate=0.01)
    model.initialize_model(data)
    model.train(data, target_words, num_epochs=1000, early_stopping_patience=50)

    print_section_header("Prediction Results")
    table_data = []
    for seq in test_sequences:
        pred_word, prob = model.predict_next_word(seq, top_k=1)
        status = "✓" if pred_word in target_words else "✗"
        table_data.append([
            ' '.join(seq),
            pred_word if pred_word else "Error",
            f"{prob:.4f}" if prob else "N/A",
            status
        ])
    print_table(
        headers=["Input Sequence", "Predicted Word", "Probability", "Correct"],
        rows=table_data
    )
    model.save_predictions(test_sequences)
    print(f"\n{' Analysis Complete '.center(50, '*')}")
if __name__ == "__main__":
    main()