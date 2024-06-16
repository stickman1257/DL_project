import torch
from tqdm.auto import tqdm
from transformers import DistilBertTokenizer, GPT2Tokenizer
import pandas as pd
from torch.utils.data import DataLoader
from src.model import VQAModel
from src.dataset import VQADataset
import warnings
import numpy as np
from sklearn.model_selection import GroupKFold
from collections import OrderedDict

warnings.filterwarnings('ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"current device is {device}")

# Load test data
test_df = pd.read_csv('./data/test2.csv')
train_img_path = './data/image/train'

# Initialize tokenizer
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
vocab_size = len(tokenizer)

# Create test dataset and dataloader
test_dataset = VQADataset(test_df, tokenizer, train_img_path, img_size=480, is_train=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

# Group K-Fold
n_splits = 100
gkf = GroupKFold(n_splits=n_splits)
groups = test_df['image_id'].values

# Placeholder for storing predictions
all_preds = []

# Perform inference using each fold's model
for fold in range(n_splits):
    print(f"Evaluating fold {fold + 1}/{n_splits}")

    # Load model
    model = VQAModel(vocab_size).to(device)
    model_path = './data/checkpoints/resgpt/92_100_2e-03_resgpt2ver1_model.pt'
    tmp_weight = torch.load(model_path)
    weight = OrderedDict()
    for k, v in tmp_weight.items():
        weight[k[7:]] = v

    model.load_state_dict(weight)
    model.eval()

    preds = []
    with torch.no_grad():
        for data in tqdm(test_loader, total=len(test_loader)):
            images = data['image'].to(device)
            question = data['question'].to(device)
            padding_mask = data['padding_mask'].to(device)

            outputs = model(images, question, padding_mask)
            _, pred = outputs.max(-1)
            preds.extend(pred.cpu().numpy())

    all_preds.append(preds)

# Hard voting ensemble
all_preds = np.array(all_preds)
ensemble_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_preds)

# Decode predictions
decoded_output = [test_dataset.label2ans[pred] for pred in ensemble_preds]

# Save predictions to CSV
sample_submission = pd.read_csv('./data/updated_sub.csv')
sample_submission['answer'] = decoded_output
sample_submission.to_csv('./ver06_gpt_ensemble.csv', index=False)

# Calculate accuracy
correct = 0
total = 0
for pred, answer in zip(decoded_output, test_df['answer']):
    if pred == answer:
        correct += 1
    total += 1

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")
