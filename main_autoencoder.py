from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

from autoencoder.network import Autoencoder
from autoencoder.train import train_autoencoder, log_latent_space_visualization
from utils.load_fashion_mnist import load_fashion_mnist

# parameters
LATENT_DIM = 64
NUM_EPOCHS = 20
LOG_LATENT_SPACE = False

if __name__ == "__main__":
    # Load data
    print("Loading Fashion-MNIST dataset...")
    train_loader, test_loader = load_fashion_mnist()

    # Create model
    model = Autoencoder(latent_dim=LATENT_DIM)
    print(f"Created autoencoder with latent dimension: {LATENT_DIM}")

    # Train model
    print("Training autoencoder...")
    train_losses, val_losses = train_autoencoder(model, train_loader, test_loader, epochs=NUM_EPOCHS)

    if LOG_LATENT_SPACE:
        # Log final latent space visualization
        print("Creating latent space visualization...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f'runs/fashion_mnist_autoencoder_{timestamp}'
        print(f"TensorBoard logs for latent space will be saved to: {log_dir}")
        print(f"To view latent space TensorBoard, run: tensorboard --logdir={log_dir}")
        writer = SummaryWriter(log_dir)
        log_latent_space_visualization(model, test_loader, writer, NUM_EPOCHS)
        writer.close()
