# Video Analysis with MELD Dataset

This project performs multimodal emotion recognition using the MELD (Multimodal EmotionLines Dataset) dataset, which contains conversations from the Friends TV series with emotion and sentiment annotations.

## Dataset Setup

1. Download the MELD dataset using either of these commands:
```bash
wget https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz
# or
wget https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz
```

2. Extract the downloaded tar.gz file:
```bash
tar -xzvf MELD.Raw.tar.gz
```

3. After extraction, you'll find several tar.gz files. Extract them as well:
```bash
cd MELD.Raw
tar -xzvf train.tar.gz
tar -xzvf dev.tar.gz
tar -xzvf test.tar.gz
```

4. Create the dataset directory structure:
```bash
# Create main dataset directory
mkdir -p dataset/{train,dev,test}

# Move files to respective directories
mv train_sent_emo.csv dataset/train/
mv train_splits dataset/train/

mv dev_sent_emo.csv dataset/dev/
mv dev_splits_complete dataset/dev/

mv test_sent_emo.csv dataset/test/
mv output_repeated_splits_test dataset/test/
```

## Project Structure
```
.
├── dataset/
│   ├── train/
│   │   ├── train_sent_emo.csv
│   │   └── train_splits/
│   ├── dev/
│   │   ├── dev_sent_emo.csv
│   │   └── dev_splits_complete/
│   └── test/
│       ├── test_sent_emo.csv
│       └── output_repeated_splits_test/
├── training/
├── deployment/
├── README.md
└── .gitignore
```