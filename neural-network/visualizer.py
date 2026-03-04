import matplotlib.pyplot as plt

## The idea is to logg and visualize the training and testing of data


class Visualizer:

    """
    Idea of an history objects
                {
                "epochs": [],              # [0,1,2,...] (index is epoch)
                "train_loss": [],          # scalar per epoch (dataset-average)
                "val_loss": [],            # scalar per epoch (optional)
                "layers": {
                    0: {                   # layer index
                        "weight_norm": [], # scalar per epoch
                        "grad_norm": [],   # scalar per epoch (||dW||)
                        # optionally: "activation_mean": [], "activation_std": []
                    },
                    1: { ... }
                }
            }
    """
    
    def __init__(self, history):
        self.history = history
        
    def plot_loss(self):
        epochs = self.history["epochs"]
        plt.figure()
        plt.plot(epochs, self.history["train_loss"], label="train")
        if self.history.get("val_loss"):
            # val_loss may be shorter or same length
            plt.plot(range(len(self.history["val_loss"])), self.history["val_loss"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs Epoch")
        plt.legend()
        plt.show()

    def plot_weight_norms(self):
        epochs = self.history["epochs"]
        plt.figure()
        for idx, layer_data in self.history["layers"].items():
            plt.plot(epochs, layer_data["weight_norm"], label=f"layer {idx}")
        plt.xlabel("Epoch")
        plt.ylabel("Weight Norm")
        plt.title("Weight Norms per Layer")
        plt.legend()
        plt.show()

    def plot_grad_norms(self):
        epochs = self.history["epochs"]
        plt.figure()
        for idx, layer_data in self.history["layers"].items():
            plt.plot(epochs, layer_data["grad_norm"], label=f"layer {idx}")
        plt.xlabel("Epoch")
        plt.ylabel("Grad Norm")
        plt.title("Gradient Norms per Layer")
        plt.legend()
        plt.show()

    def plot_loss_vs_weight(self, layer_idx):
        # Loss on y, weight_norm on x
        weight_norm = self.history["layers"][layer_idx]["weight_norm"]
        loss = self.history["train_loss"]
        plt.figure()
        plt.scatter(weight_norm, loss)
        plt.xlabel(f"Layer {layer_idx} Weight Norm")
        plt.ylabel("Train Loss")
        plt.title(f"Loss vs Weight Norm (Layer {layer_idx})")
        plt.show()