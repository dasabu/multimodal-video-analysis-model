import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models
from sklearn.metrics import precision_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

from meld_dataset import MELDDataset

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

class MultimodalTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Log dataset sized
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)

        print(f'Training size: {train_size:,} samples')
        print(f'Val size: {val_size:,} samples')
        print(f'Batch per epoch: {len(train_loader):,}')

        # Tensorboard logging
        timestamp = datetime.now().strftime('%b%d_%H-%M-%S') # Dec17_14-22-35
        # When upload training script to SageMaker, it will inject SM_MODEL_DIR to environment variable
        base_dir = '/opt/ml/output/tensorboard' if 'SM_MODEL_DIR' in os.environ else 'runs'
        log_dir = f'{base_dir}/run_{timestamp}'

        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0 # current epoch

        self.optimizer = torch.optim.Adam([
            {'params': self.model.text_encoder.parameters(), 'lr': 8e-6},
            {'params': self.model.video_encoder.parameters(), 'lr': 8e-5},
            {'params': self.model.audio_encoder.parameters(), 'lr': 8e-5},
            {'params': self.model.fusion_layer.parameters(), 'lr': 5e-4},
            {'params': self.model.emotion_classifier.parameters(), 'lr': 5e-4},
            {'params': self.model.sentiment_classifier.parameters(), 'lr': 5e-4}
        ], weight_decay=1e-5)

        # Reduce learning rate when loss stops decreasing: multiply by 0.1 after 2 epochs without improvement
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode='min',
            factor=0.1,
            patience=2,
        )

        # label_smoothing: add small value to the true label to make it less confident to avoid overfitting
        # [0, 1, 0] -> [0.05, 0.9, 0.05]
        self.emotion_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.sentiment_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        self.current_train_losses = None # for log_metrics
    
    def train_epoch(self):
        self.model.train()
        running_losses = { 'total': 0, 'emotion': 0, 'sentiment': 0 }

        for batch in self.train_loader:
            device = next(self.model.parameters()).device # get device from model parameters

            text_inputs = {
                'input_ids': batch['text_inputs']['input_ids'].to(device),
                'attention_mask': batch['text_inputs']['attention_mask'].to(device)
            }
            video_frames = batch['video_frames'].to(device)
            audio_features = batch['audio_features'].to(device)
            emotion_labels = batch['emotion_labels'].to(device)
            sentiment_labels = batch['sentiment_labels'].to(device)

            # Zero gradients (reset) before each batch
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(text_inputs, video_frames, audio_features)

            # Calculate losses using raw logits (no softmax)
            emotion_loss = self.emotion_criterion(outputs['emotions'], emotion_labels)
            sentiment_loss = self.sentiment_criterion(outputs['sentiments'], sentiment_labels)

            # Total loss
            total_loss = emotion_loss + sentiment_loss

            # Backward pass, calculate gradients
            total_loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update model parameters
            self.optimizer.step()

            # Track losses
            running_losses['total'] += total_loss.item()
            running_losses['emotion'] += emotion_loss.item()
            running_losses['sentiment'] += sentiment_loss.item()

            self.log_metrics({
                'total': total_loss.item(),
                'emotion': emotion_loss.item(),
                'sentiment': sentiment_loss.item()
            })

        self.global_step += 1

        return {k: v / len(self.train_loader) for k, v in running_losses.items()} # average loss per batch

    # Validation and testing
    def evaluate(self, data_loader, phase='val'):
        self.model.eval()
        losses = { 'total': 0, 'emotion': 0, 'sentiment': 0 }

        all_emotion_preds = []
        all_emotion_labels = []

        all_sentiment_preds = []
        all_sentiment_labels = []

        with torch.inference_mode():
            for batch in data_loader:
                device = next(self.model.parameters()).device

                text_inputs = {
                    'input_ids': batch['text_inputs']['input_ids'].to(device),
                    'attention_mask': batch['text_inputs']['attention_mask'].to(device)
                }

                video_frames = batch['video_frames'].to(device)
                audio_features = batch['audio_features'].to(device)

                emotion_labels = batch['emotion_labels'].to(device)
                sentiment_labels = batch['sentiment_labels'].to(device)

                # Forward pass
                outputs = self.model(text_inputs, video_frames, audio_features)

                # Calculate losses
                emotion_loss = self.emotion_criterion(outputs['emotions'], emotion_labels)
                sentiment_loss = self.sentiment_criterion(outputs['sentiments'], sentiment_labels)
                total_loss = emotion_loss + sentiment_loss

                # argmax: get the index of the highest probability
                # cpu(): move to CPU
                # numpy(): convert to numpy array
                # extend: add to the end of the list
                all_emotion_preds.extend(outputs['emotions'].argmax(dim=1).cpu().numpy())
                all_emotion_labels.extend(emotion_labels.cpu().numpy())
                all_sentiment_preds.extend(outputs['sentiments'].argmax(dim=1).cpu().numpy())
                all_sentiment_labels.extend(sentiment_labels.cpu().numpy())

                # Track losses
                losses['total'] += total_loss.item()
                losses['emotion'] += emotion_loss.item()
                losses['sentiment'] += sentiment_loss.item()

        # Average loss per batch
        avg_loss = {k: v / len(data_loader) for k, v in losses.items()}

        # Compute the precision and accuracy
        emotion_precision = precision_score(all_emotion_labels, all_emotion_preds, average='weighted')
        emotion_accuracy = accuracy_score(all_emotion_labels, all_emotion_preds)

        sentiment_precision = precision_score(all_sentiment_labels, all_sentiment_preds, average='weighted')
        sentiment_accuracy = accuracy_score(all_sentiment_labels, all_sentiment_preds)
        
        self.log_metrics(avg_loss, {
            'emotion_precision': emotion_precision,
            'emotion_accuracy': emotion_accuracy,
            'sentiment_precision': sentiment_precision,
            'sentiment_accuracy': sentiment_accuracy
        }, phase)

        if phase == 'val':
            # Reduce learning rate when loss stops decreasing
            self.scheduler.step(avg_loss['total'])

        return avg_loss, {
            'emotion_precision': emotion_precision,
            'emotion_accuracy': emotion_accuracy,
            'sentiment_precision': sentiment_precision,
            'sentiment_accuracy': sentiment_accuracy
        }

    def log_metrics(self, losses, metrics=None, phase='train'):
        if phase == 'train':
            self.current_train_losses = losses
        # Validation phase
        else:
            # add metrics to tensorboard to visualize as a line graph
            self.writer.add_scalar('loss/total/train', self.current_train_losses['total'], self.global_step)
            self.writer.add_scalar('loss/total/val', losses['total'], self.global_step)

            self.writer.add_scalar('loss/emotion/train', self.current_train_losses['emotion'], self.global_step)
            self.writer.add_scalar('loss/emotion/val', losses['emotion'], self.global_step)

            self.writer.add_scalar('loss/sentiment/train', self.current_train_losses['sentiment'], self.global_step)
            self.writer.add_scalar('loss/sentiment/val', losses['sentiment'], self.global_step)

        if metrics:
            self.writer.add_scalar(f'{phase}/emotion_precision', metrics['emotion_precision'], self.global_step)
            self.writer.add_scalar(f'{phase}/emotion_accuracy', metrics['emotion_accuracy'], self.global_step)
            
            self.writer.add_scalar(f'{phase}/sentiment_precision', metrics['sentiment_precision'], self.global_step)
            self.writer.add_scalar(f'{phase}/sentiment_accuracy', metrics['sentiment_accuracy'], self.global_step)

if __name__ == '__main__':
    dataset = MELDDataset(
        '../dataset/train/train_sent_emo.csv',
        '../dataset/train/train_splits',
    )

    sample = dataset[0]

    model = MultimodalModel()

    model.eval() # evaluation mode (not training)

    text_inputs = {
        'input_ids': sample['text_inputs']['input_ids'].unsqueeze(0),
        'attention_mask': sample['text_inputs']['attention_mask'].unsqueeze(0)
    }

    video_frames = sample['video_frames'].unsqueeze(0)  
    audio_features = sample['audio_features'].unsqueeze(0)

    with torch.inference_mode(): # no_grad()
        outputs = model(text_inputs, video_frames, audio_features)
        
        emotion_probs = torch.softmax(outputs['emotions'], dim=1)[0]
        sentiment_probs = torch.softmax(outputs['sentiments'], dim=1)[0]

    emotion_map = {
        0: 'anger',
        1: 'disgust',
        2: 'fear',
        3: 'joy',
        4: 'neutral',
        5: 'sadness',
        6: 'surprise'
    }

    sentiment_map = {
        0: 'negative',
        1: 'neutral',
        2: 'positive'
    }

    for i, prob in enumerate(emotion_probs):
        print(f'{emotion_map[i]}: {prob:.4f}')

    for i, prob in enumerate(sentiment_probs):
        print(f'{sentiment_map[i]}: {prob:.4f}')