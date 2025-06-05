from feedforward_nn.load_fashion_mnist import load_fashion_mnist
from feedforward_nn.network import FeedforwardNN, SimpleNN, DeepNN
from feedforward_nn.train import train_model

if __name__ == "__main__":
    print("Loading Fashion-MNIST dataset...")
    train_loader, val_loader, test_loader = load_fashion_mnist(batch_size=128)

    print("Creating model...")

    # Option 1: Default architecture
    # model = FeedforwardNN(hidden_sizes=[512, 256, 128], dropout_rate=0.3, use_batch_norm=True)

    # Option 2: Simple architecture
    # model = SimpleNN()

    # Option 3: Deep architecture
    model = DeepNN()

    print(f"Model architecture:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Train model
    print("Training model...")
    model, train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader, val_loader, test_loader, epochs=20, lr=1e-3
    )

    print("Done :)")