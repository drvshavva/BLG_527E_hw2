import torch.nn as nn


class FeedforwardNN(nn.Module):
    """Feedforward Neural Network for Fashion-MNIST Classification"""

    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128],
                 num_classes=10, dropout_rate=0.3, use_batch_norm=True):
        super(FeedforwardNN, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Build layers dynamically
        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))

            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

            # Activation
            layers.append(nn.ReLU())

            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        return self.network(x)


class SimpleNN(nn.Module):
    """Simple 2-layer network"""

    def __init__(self):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


class DeepNN(nn.Module):
    """Deep network with 5 hidden layers"""

    def __init__(self):
        super(DeepNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)
