import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Fashion-MNIST class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def train_model(model, train_loader, val_loader, test_loader, epochs=50, lr=1e-3, weight_decay=1e-4):
    """Train the feedforward neural network"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    # Setup TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'runs/fashion_mnist_ffnn_{timestamp}'
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")
    print(f"To view TensorBoard, run: tensorboard --logdir={log_dir}")

    model.to(device)

    # Log model architecture
    sample_input = torch.randn(1, 1, 28, 28).to(device)
    writer.add_graph(model, sample_input)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

            # Log batch metrics
            if batch_idx % 100 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/Batch_Train', loss.item(), global_step)
                writer.add_scalar('Accuracy/Batch_Train',
                                  100. * (predicted == target).sum().item() / target.size(0),
                                  global_step)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        # Calculate epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total

        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        # Log epoch metrics
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        print(f'Epoch [{epoch + 1}/{epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 60)

    # Load best model
    model.load_state_dict(best_model_state)
    comprehensive_evaluation_with_tensorboard(
        model, test_loader, class_names, writer, epochs
    )
    writer.close()

    return model, train_losses, train_accuracies, val_losses, val_accuracies


def evaluate_model(model, test_loader, writer, epoch=0):
    """Evaluate model on test dataset and log to TensorBoard"""
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)

            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    test_accuracy = 100. * test_correct / test_total
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    writer.add_scalar('Test/Accuracy', test_accuracy, epoch)
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    return test_accuracy, all_predictions, all_targets


def plot_confusion_matrix(y_true, y_pred, class_names, writer, epoch=0):
    """Generate confusion matrix and log to TensorBoard"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Figure'ı TensorBoard'a kaydet
    writer.add_figure('Confusion Matrix', plt.gcf(), epoch)
    plt.close()

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    writer.add_figure('Normalized Confusion Matrix', plt.gcf(), epoch)
    plt.close()

    return cm


def show_misclassified_examples(model, test_loader, class_names, writer,
                                epoch=0, num_examples=12):
    """Log misclassified examples to TensorBoard"""
    model.eval()
    misclassified = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)

            mask = predicted != target
            if mask.sum() > 0:
                misclassified_data = data[mask]
                misclassified_targets = target[mask]
                misclassified_preds = predicted[mask]

                for i in range(min(len(misclassified_data), num_examples - len(misclassified))):
                    misclassified.append({
                        'image': misclassified_data[i].cpu(),
                        'true': misclassified_targets[i].cpu().item(),
                        'pred': misclassified_preds[i].cpu().item()
                    })

                if len(misclassified) >= num_examples:
                    break

    if misclassified:
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        axes = axes.flatten()

        for i, example in enumerate(misclassified[:num_examples]):
            img = example['image']

            # Grayscale için format düzelt
            if len(img.shape) == 3 and img.shape[0] == 1:
                img = img.squeeze(0)  # (1, H, W) -> (H, W)

            if img.min() < 0:  # Normalize edilmişse denormalize et
                img = (img + 1) / 2

            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'True: {class_names[example["true"]]}\n'
                              f'Pred: {class_names[example["pred"]]}')
            axes[i].axis('off')

        plt.suptitle('Misclassified Examples')
        plt.tight_layout()
        writer.add_figure('Misclassified Examples', fig, epoch)
        plt.close()

        # Tek tek de kaydet
        for i, example in enumerate(misclassified[:min(8, num_examples)]):
            img = example['image']

            if len(img.shape) == 3 and img.shape[0] == 1:
                img = img.squeeze(0)  # (1, H, W) -> (H, W)

            if img.min() < 0:
                img = (img + 1) / 2

            writer.add_image(
                f'Misclassified/True_{class_names[example["true"]]}_Pred_{class_names[example["pred"]]}',
                img, epoch, dataformats='HW'
            )


def log_classification_report_to_tensorboard(y_true, y_pred, class_names, writer, epoch=0):
    """Log classification report to TensorBoard"""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    for class_name in class_names:
        if class_name in report:
            writer.add_scalar(f'ClassMetrics/{class_name}/Precision',
                              report[class_name]['precision'], epoch)
            writer.add_scalar(f'ClassMetrics/{class_name}/Recall',
                              report[class_name]['recall'], epoch)
            writer.add_scalar(f'ClassMetrics/{class_name}/F1-Score',
                              report[class_name]['f1-score'], epoch)

    writer.add_scalar('Overall/Precision', report['macro avg']['precision'], epoch)
    writer.add_scalar('Overall/Recall', report['macro avg']['recall'], epoch)
    writer.add_scalar('Overall/F1-Score', report['macro avg']['f1-score'], epoch)
    writer.add_scalar('Overall/Weighted-F1', report['weighted avg']['f1-score'], epoch)

    report_text = classification_report(y_true, y_pred, target_names=class_names)
    writer.add_text('Classification Report', report_text.replace('\n', '  \n'), epoch)

    print("Classification Report:")
    print("=" * 60)
    print(report_text)


def comprehensive_evaluation_with_tensorboard(model, test_loader, class_names,
                                              writer, epoch=0):
    """Comprehensive model evaluation with TensorBoard logging"""
    print(f"Running comprehensive evaluation for epoch {epoch}...")

    test_accuracy, all_predictions, all_targets = evaluate_model(
        model, test_loader, writer, epoch
    )

    plot_confusion_matrix(
        all_targets, all_predictions, class_names, writer, epoch
    )

    show_misclassified_examples(
        model, test_loader, class_names, writer, epoch
    )

    log_classification_report_to_tensorboard(
        all_targets, all_predictions, class_names, writer, epoch
    )

    writer.flush()
    print(f"All evaluation results logged to TensorBoard for epoch {epoch}")
