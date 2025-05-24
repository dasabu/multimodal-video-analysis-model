import torch
import torchaudio
import os
import cv2
import numpy as np
import subprocess
import whisper
from transformers import AutoTokenizer

from models import MultimodalModel

EMOTION_MAP = {
    0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'neutral', 5: 'sadness', 6: 'surprise'
}

SENTIMENT_MAP = {
    0: 'negative', 1: 'neutral', 2: 'positive'
}

class VideoProcessor:
    def process_video(self, video_path):
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

class AudioProcessor:
    def extract_features(self, video_path):
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

class VideoUtteranceProcessor:
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()

    def extract_segment(self, video_path, start_time, end_time, tmp_dir='/tmp'):
        os.makedirs(tmp_dir, exist_ok=True)
        segment_path = os.path.join(tmp_dir, f'segment_{start_time}_{end_time}.mp4')

        subprocess.run([
            'ffmpeg', '-i', video_path,
            '-ss', str(start_time),
            '-to', str(end_time),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-y',
            segment_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not os.path.exists(segment_path) or os.path.getsize(segment_path) == 0:
            raise ValueError(f"Failed to extract segment ({segment_path}) from {video_path} between {start_time} and {end_time}")

        return segment_path

        
def model_fn(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MultimodalModel().to(device)
    model_path = os.path.join(model_dir, 'model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found at {model_path}')

    print(f'Loading model from {model_path}...')
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    return {
        'model': model,
        'device': device,
        'tokenizer': AutoTokenizer.from_pretrained('bert-base-uncased'),
        'transcriber': whisper.load_model('base', device='cpu' if device.type == 'cpu' else device)
    }

def predict_fn(input_data, model_dict):
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']
    device = model_dict['device']

    video_path = input_data['video_path']
    result = model_dict['transcriber'].transcribe(video_path, word_timestamps=True)

    utterance_processor = VideoUtteranceProcessor()
    predictions = []

    for segment in result['segments']:
        try:
            segment_path = utterance_processor.extract_segment(video_path, segment['start'], segment['end'])

            video_frames = utterance_processor.video_processor.process_video(segment_path)
            audio_features   = utterance_processor.audio_processor.extract_features(segment_path)
            text_inputs = tokenizer(
                segment['text'],
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(device)

            # Move to device
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            # Add a dimension for batch size
            video_frames = video_frames.unsqueeze(0).to(device)
            audio_features = audio_features.unsqueeze(0).to(device)

            # Get prediction

            with torch.inference_mode():
                outputs = model(text_inputs, video_frames, audio_features)
                # Softmax of 1st dimension (0th is batch dimension)
                emotion_probs = torch.softmax(outputs['emotion'], dim=1)[0]
                sentiment_probs = torch.softmax(outputs['sentiment'], dim=1)[0]
                # Top 3
                emotion_values, emotion_indices = torch.topk(emotion_probs, k=3)
                sentiment_values, sentiment_indices = torch.topk(sentiment_probs, k=3)
            
            predictions.append({
                'start_time': segment['start'],
                'end_time': segment['end'],
                'text': segment['text'],
                'emotions': [
                    { 'label': EMOTION_MAP[i.item()], 'confidence': v.item() } for i, v in zip(emotion_indices, emotion_values)
                ],
                'sentiment': [
                    { 'label': SENTIMENT_MAP[i.item()], 'confidence': v.item() } for i, v in zip(sentiment_indices, sentiment_values)
                ]
            })

        except Exception as e:
            print(f'Segment inference failed: {e}')
        finally:
            # Clean up
            if os.path.exists(segment_path):
                os.remove(segment_path)

    return { 'utterances': predictions }

def process_local_video(video_path, model_dir='model'):
    model_dict = model_fn(model_dir)

    input_data = { 'video_path': video_path }

    predictions = predict_fn(input_data, model_dict)

    for utterance in predictions['utterances']:
        print(f'{utterance['start_time']} - {utterance['end_time']}: {utterance['text']}')
        print(f'Emotions:')
        for emotion in utterance['emotions']:
            print(f'- {emotion['label']}: {emotion['confidence']:.4f}')
        print(f'Sentiment:')
        for sentiment in utterance['sentiment']:
            print(f'- {sentiment['label']}: {sentiment['confidence']:.4f}')
        print('-' * 50)
