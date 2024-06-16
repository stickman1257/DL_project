import os
import pandas as pd

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models # 이미지
from torchvision import transforms

from PIL import Image

from transformers import GPT2Tokenizer, GPT2Model # 텍스트

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"current device is {device}")

class VQADataset(Dataset):
    def __init__(self, df, tokenizer, transform, img_path, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.transform = transform
        self.img_path = img_path
        self.is_test = is_test


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = os.path.join(self.img_path, row['image_id'] + '.jpg') # 이미지
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)

        question = row['question'] # 질문
        question = self.tokenizer.encode_plus(
            question,
            truncation=True,
            add_special_tokens=True,
            max_length=32,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        if not self.is_test:
            answer = row['answer'] # 답변
            answer = self.tokenizer.encode_plus(
                answer,
                max_length=32,
                padding='max_length',
                truncation=True,
                return_tensors='pt')
            return {
                'image': image.squeeze(),
                'question': question['input_ids'].squeeze(),
                'answer': answer['input_ids'].squeeze()
            }
        else:
            return {
                'image': image,
                'question': question['input_ids'].squeeze(),
            }
            
class VQAModel(nn.Module):
    def __init__(self, vocab_size):
        super(VQAModel, self).__init__()
        self.vocab_size = vocab_size

        self.resnet = models.resnet50(pretrained=True)
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.gpt2.resize_token_embeddings(vocab_size) # 추가한 [PAD] 토큰 반영

        combined_features_size = 1000 + self.gpt2.config.hidden_size # resnet 출력 차원 + gpt2 출력 차원
        self.classifier = nn.Linear(combined_features_size, vocab_size)

    def forward(self, images, question):
        image_features = self.resnet(images)
        image_features = image_features.view(image_features.size(0),-1)

        outputs = self.gpt2(question)
        output_features = outputs.last_hidden_state # [batch, sequence, hidden]

        image_features = image_features.unsqueeze(1).expand(-1, output_features.size(1),-1) # [batch, sequence, 1000]

        combined = torch.cat([image_features, output_features], dim=-1)
        print(combined) # [batch, sequence, 1000+hidden]
        output = self.classifier(combined)
        print(output) # [batch, vocab_size]
        return output
    
    
# 데이터 불러오기
train_df = pd.read_csv('../data/train.csv')
#test_df = pd.read_csv('../data/test.csv')
sample_submission = pd.read_csv('../data/sample_submission.csv')
train_img_path = '../data/image/train'
#test_img_path = '../data/image/test'

# dataset & dataloader
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
vocab_size = len(tokenizer)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

from sklearn.model_selection import train_test_split

# 데이터 나누기
train_data, test_data = train_test_split(train_df, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.125, random_state=42)

# 훈련 데이터셋
train_dataset = VQADataset(train_data, tokenizer, transform, train_img_path, is_test=False)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 검증 데이터셋
val_dataset = VQADataset(val_data, tokenizer, transform, train_img_path, is_test=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 테스트 데이터셋
test_dataset = VQADataset(test_data, tokenizer, transform, train_img_path, is_test=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def inference(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for data in tqdm(loader, total=len(loader)):
            images = data['image'].to(device)
            question = data['question'].to(device)

            outputs = model(images, question) # [batch, sequence, vocab]

            _, pred = torch.max(outputs, dim=2) # values, indices = _, pred
            preds.extend(pred.cpu().numpy())

    return preds

# Model
model = VQAModel(vocab_size).to(device)

# Criterion and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in tqdm(train_loader, total=len(train_loader)):
        images = data['image'].to(device)
        question = data['question'].to(device)
        answers = data['answer'].to(device)

        optimizer.zero_grad()
        outputs = model(images, question)
        loss = criterion(outputs.view(-1, outputs.size(-1)), answers.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(val_loader, total=len(val_loader)):
            images = data['image'].to(device)
            question = data['question'].to(device)
            answers = data['answer'].to(device)

            outputs = model(images, question)
            loss = criterion(outputs.view(-1, outputs.size(-1)), answers.view(-1))

            total_loss += loss.item()
    return total_loss / len(val_loader)
# Training loop
for epoch in range(1):
    avg_train_loss = train(model, train_loader, optimizer, criterion)
    avg_val_loss = evaluate(model, val_loader, criterion)
    print(f"Epoch: {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")



# inference
preds = inference(model, test_loader)

no_pad_output = []
for pred in preds:
    output = pred[pred != 50257] # [PAD] token 제외
    no_pad_output.append(tokenizer.decode(output).strip())



# 정확도 계산
correct = 0
total = 0

for pred, answer in zip(no_pad_output, test_data['answer']):
    if pred == answer:
        correct += 1
    total += 1

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")
                         
