import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from datetime import datetime
import torch
import torchvision
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 3. Training Function
def train_autoencoder(model, train_loader, test_loader, epochs=20, lr=1e-3, log_dir=None):
    """Train the autoencoder with TensorBoard logging"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Setup TensorBoard
    if log_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f'runs/fashion_mnist_autoencoder_{timestamp}'

    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"To view TensorBoard, run: tensorboard --logdir={log_dir}")

    train_losses = []
    val_losses = []

    model.to(device)

    # Log model architecture
    sample_input = torch.randn(1, 1, 28, 28).to(device)
    writer.add_graph(model, sample_input)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            reconstructed = model(data)
            loss = criterion(reconstructed, data)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Log batch loss every 100 batches
            if batch_idx % 100 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/Batch_Train', loss.item(), global_step)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                reconstructed = model(data)
                val_loss += criterion(reconstructed, data).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Log epoch losses
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

        # Log sample reconstructions every 5 epochs
        if epoch % 5 == 0:
            log_sample_reconstructions(model, test_loader, writer, epoch, criterion=criterion)

        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # Final Reconstruction Loss Log
    writer.add_scalar('Final_Loss/Train', train_losses[-1], epochs)
    writer.add_scalar('Final_Loss/Validation', val_losses[-1], epochs)

    # Log final generated samples
    log_generated_samples(model, writer, epochs)

    writer.close()
    return train_losses, val_losses


def log_sample_reconstructions(model, test_loader, writer, epoch, num_images=8, criterion=None):
    """Log sample reconstructions to TensorBoard"""
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data[:num_images].to(device)
        reconstructed = model(data)

        # Create comparison grid (data already in [0,1])
        comparison = torch.cat([data, reconstructed], dim=0)

        # Log to TensorBoard
        img_grid = torchvision.utils.make_grid(comparison, nrow=num_images, normalize=False)
        writer.add_image(f'Reconstructions/Epoch_{epoch}', img_grid, epoch)

        if criterion is not None:
            recon_loss = criterion(reconstructed, data).item()
            writer.add_scalar(f'Reconstruction/Sample_Batch_Loss_{epoch}', recon_loss, epoch)


def log_generated_samples(model, writer, epoch, num_samples=16):
    """Log generated samples to TensorBoard"""
    model.eval()
    with torch.no_grad():
        # Sample from latent space (data already in [0,1])
        z = torch.randn(num_samples, model.latent_dim).to(device)
        generated = model.decode(z)

        generated_upscaled = F.interpolate(generated, scale_factor=4, mode='bilinear', align_corners=False)

        # Log to TensorBoard
        img_grid = torchvision.utils.make_grid(generated_upscaled, nrow=4, normalize=False)
        writer.add_image('Generated_Samples', img_grid, epoch)


def log_latent_space_visualization(model, test_loader, writer, epoch, num_samples=1000):
    """Log latent space visualization to TensorBoard"""
    model.eval()
    latent_vectors = []
    labels = []

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if len(latent_vectors) >= num_samples:
                break

            data = data.to(device)
            latent = model.encode(data)

            latent_vectors.append(latent.cpu())
            labels.extend(target.cpu().numpy())

    latent_vectors = torch.cat(latent_vectors)[:num_samples]
    labels = np.array(labels)[:num_samples]

    # Fashion-MNIST class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Create metadata for visualization
    metadata = [class_names[label] for label in labels]

    writer.add_embedding(latent_vectors, metadata=metadata, global_step=epoch)
