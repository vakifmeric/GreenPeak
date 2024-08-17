import torch
import torch.nn as nn


class GreenPeakModel(nn.Module):
    def __init__(self, num_classes=1):
        super(GreenPeakModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=50, out_channels=32, kernel_size=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=2),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.lstm = nn.LSTM(input_size=32, hidden_size=128, num_layers=2, batch_first=True)


        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        x = self.cnn(x)

        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]

        x = self.fc(x)
        return x
