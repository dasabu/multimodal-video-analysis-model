import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Not trainable (only use)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.projection = nn.Linear(768, 128) # 768 = self.bert output size
    
    def forward(self, input_ids, attention_mask):
        # Extract bert embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token for classification
        # bert can train on [CLS] token to capture the meaning of the whole sentence
        pooler_output = outputs.pooler_output

        return self.projection(pooler_output)
    
class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vision_models.video.r3d_18(pretrained=True)

        # Not trainable (only use)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Number of features (node) from the last layer
        num_features = self.backbone.fc.in_features
        # Replace the last layer with a new linear layer
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 128), # Only this layer is trainable
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        # Before: [batch_size, frames, channels, height, width] 
        # After: [batch_size, channels, frames, height, width]
        x = x.transpose(1, 2)

        return self.backbone(x)

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Lower level features
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Higher level features
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        for params in self.conv_layers.parameters():
            params.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        # Remove the single channel dimension
        # [batch_size, 1, 64, 300] -> [batch_size, 64, 300]
        # this will suitable to the first conv layer (64, 64, kernel_size=3)
        x = x.squeeze(1)

        features = self.conv_layers(x)
        # Features shape: [batch_size, 128, 1]
        # 1 is from the adaptive avg pool
        # Squeeze the last dimension to get [batch_size, 128]

        return self.projection(features.squeeze(-1))
        # Projection shape: [batch_size, 128]

class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder() # shape = [batch_size, 128]
        self.video_encoder = VideoEncoder() # shape = [batch_size, 128]
        self.audio_encoder = AudioEncoder() # shape = [batch_size, 128]

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        ) # shape = [batch_size, 256]

        # Classification layer
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7)
        ) # shape = [batch_size, 7]

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        ) # shape = [batch_size, 3]

    def forward(self, text_inputs, video_frames, audio_features):
        _text_features = self.text_encoder(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask']
        )

        _video_features = self.video_encoder(video_frames)
        _audio_features = self.audio_encoder(audio_features)

        # Concatenate multimodal features
        combined_features = torch.cat([
            _text_features, 
            _video_features,
            _audio_features
        ], dim=1) # shape = [batch_size, 128 * 3]

        # Fusion layer
        fused_features = self.fusion_layer(combined_features)

        # Classification layers
        emotion_output = self.emotion_classifier(fused_features)
        sentiment_output = self.sentiment_classifier(fused_features)

        return {
            'emotions': emotion_output,
            'sentiments': sentiment_output
        }