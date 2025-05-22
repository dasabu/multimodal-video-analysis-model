import torch
import torch.utils.data.dataloader
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
import os
import cv2
import numpy as np
import subprocess

os.environ['TOKENIZERS_PARALLELISM'] = "false"

class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        self.emotion_map = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'joy': 3,
            'neutral': 4,
            'sadness': 5,
            'surprise': 6
        }

        self.sentiment_map = {
            'negative': 0,
            'neutral': 1,
            'positive': 2
        }
    
    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            if not cap.isOpened():
                raise IOError(f"Could not open video file: {video_path}")

            # Read the first frame to validate the video
            ret, frame = cap.read()
            if not ret or frame is None:
                raise IOError(f"Error reading video file: {video_path}")

            # Reset index to not skip the first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while len(frames) < 30:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame = cv2.resize(frame, (224, 224))
                frame = frame.astype(np.float32)
                frame /= 255.0  # normalize each RGB to [0, 1]
                frames.append(frame)

        except Exception as e:
            raise IOError(f"Error loading video frames: {e}")
        finally:
            cap.release()

        if len(frames) == 0:
            raise IOError(f"No frames could be loaded from video: {video_path}")
        
        # Pad or truncate to 30 frames
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]
        
        # Before permute: [frames, height, width, channels]
        # After permute: [frames, channels, height, width]
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)
    
    def _extract_audio_features(self, video_path):
        audio_path = video_path.replace('.mp4', '.wav')

        try:
            subprocess.run([
                'ffmpeg',
                '-i', video_path,
                '-vn', # No video, only audio
                '-acodec', 'pcm_s16le', # 16-bit PCM audio
                '-ar', '16000', # 16kHz sample rate
                '-ac', '1', # 1 channel
                audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
            
            # Create a Mel spectrogram
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512,
            )

            mel_spec= mel_spectrogram(waveform)

            # Normalize the mel spectrogram
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std())

            # Pad or truncate to 300 timesteps
            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :300]

            return mel_spec
        
        except subprocess.CalledProcessError as e:
            raise IOError(f"Error extracting audio features: {e}")

        except Exception as e:
            raise IOError(f"Error processing audio file: {e}")
        
        finally:
            # Remove the audio file after processing
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        row = self.data.iloc[idx]
        
        try:
            video_filename = f'dia{row["Dialogue_ID"]}_utt{row["Utterance_ID"]}.mp4'
            video_path = os.path.join(self.video_dir, video_filename)

            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            text_inputs = self.tokenizer(
                row['Utterance'],
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            video_frames = self._load_video_frames(video_path=video_path)
            audio_features = self._extract_audio_features(video_path=video_path)

            # Map sentiment and emotion labels
            emotion_label = self.emotion_map[row['Emotion'].lower()]
            sentiment_label = self.sentiment_map[row['Sentiment'].lower()]

            return {
                'text_inputs': {
                    'input_ids': text_inputs['input_ids'].squeeze(),
                    'attention_mask': text_inputs['attention_mask'].squeeze()
                },
                'video_frames': video_frames,
                'audio_features': audio_features,
                'emotion_label': torch.tensor(emotion_label),
                'sentiment_label': torch.tensor(sentiment_label)
            }
        except Exception as e:
            print(f"Error processing item {idx} with path {video_path}: {e}")
            return None

def collate_fn(batch):
    # Filter out None samples
    batch = list(filter(None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def prepare_dataloaders(
    train_csv, train_video_dir, 
    dev_csv, dev_video_dir,
    test_csv, test_video_dir,
    batch_size=32
):
    train_dataset = MELDDataset(train_csv, train_video_dir)
    dev_dataset = MELDDataset(dev_csv, dev_video_dir)
    test_dataset = MELDDataset(test_csv, test_video_dir)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, dev_loader, test_loader

if __name__ == "__main__":
    train_loader, dev_loader, test_loader = prepare_dataloaders(
        train_csv='../dataset/train/train_sent_emo.csv',
        train_video_dir='../dataset/train/train_splits',
        dev_csv='../dataset/dev/dev_sent_emo.csv',
        dev_video_dir='../dataset/dev/dev_splits_complete',
        test_csv='../dataset/test/test_sent_emo.csv',
        test_video_dir='../dataset/test/output_repeated_splits_test'
    )

    for batch in train_loader:
        print(batch['text_inputs']['input_ids'].shape)
        print(batch['video_frames'].shape)
        print(batch['audio_features'].shape)
        print(batch['emotion_label'])
        print(batch['sentiment_label'])
        break