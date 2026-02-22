import torch.nn as nn 
import torch.nn.functional as F 

class CILModel(nn.Module):
    def __init__(self, input_dim, feature_dim=128):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

        self.classifier = None  # SOLO para entrenamiento

    def forward(self, x):
        feats = self.feature_extractor(x)
        feats = F.normalize(feats, dim=1)

        if self.classifier is not None:
            logits = self.classifier(feats)
            return logits, feats

        return feats

    def update_classifier(self, num_classes):
        old = self.classifier
        new = nn.Linear(self.feature_extractor[-1].out_features, num_classes)

        if old is not None:
            new.weight.data[:old.out_features] = old.weight.data
            new.bias.data[:old.out_features] = old.bias.data

        self.classifier = new

# Margarita
class IDSNet(nn.Module):
    def __init__(self, input_dim, feature_dim=128):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

        self.classifier = None

    def forward(self, x):
        feats = self.feature_extractor(x)

        logits = None
        if self.classifier is not None:
            logits = self.classifier(feats)

        return logits, feats

    def update_classifier(self, num_classes):
        self.classifier = nn.Linear(
            self.feature_extractor[-1].out_features,
            num_classes
        )