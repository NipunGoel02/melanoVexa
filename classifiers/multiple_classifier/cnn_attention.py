import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()

        self.attention_weights = nn.Parameter(
            torch.randn(feature_dim)
        )

    def forward(self, x):
        weights = F.softmax(
            self.attention_weights,
            dim=0
        )

        return torch.mul(x, weights)


class AttentionClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size1=1024,
        hidden_size2=512,
        num_classes=3
    ):

        super(AttentionClassifier, self).__init__()

        self.fc1 = nn.Linear(
            input_size,
            hidden_size1
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.4)

        # Proposed attention
        self.attention = Attention(
            hidden_size1
        )

        self.fc2 = nn.Linear(
            hidden_size1,
            hidden_size2
        )

        self.fc3 = nn.Linear(
            hidden_size2,
            num_classes
        )

    def forward(self, x):

        out = self.fc1(x)

        out = self.relu(out)

        # ACTIVE ATTENTION
        out = self.attention(out)

        out = self.dropout(out)

        out = self.fc2(out)

        out = self.relu(out)

        out = self.dropout(out)

        out = self.fc3(out)

        return out