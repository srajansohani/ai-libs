import sys
import numpy as np
class Utils:


    @staticmethod
    def show_progress(current, total, batch_number=None):
        percent = (current / total) * 100
        bar_length = 30
        filled_length = int(bar_length * current // total)
        bar = "#" * filled_length + "-" * (bar_length - filled_length)

        if batch_number is not None:
            sys.stdout.write(f"\rBatch {batch_number} [{bar}] {percent:.2f}%")
        else:
            sys.stdout.write(f"\r[{bar}] {percent:.2f}%")

        sys.stdout.flush()

        if current == total:
            print()
        
    def create_batches(X, y, batch_size, shuffle=True):
        n_samples = X.shape[0]
        print(X.shape[0])
        indices = np.arange(n_samples)

        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            yield X[batch_indices], y[batch_indices]