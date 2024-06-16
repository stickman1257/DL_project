import torch
import torch.optim as optim
import torch.nn as nn
from transformers import GPT2Tokenizer
from tqdm.auto import tqdm
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from src.model import VQAModel
from src.dataset import VQADataset


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"current device is {device}")

def prepare_data(train_df, train_img_path, tokenizer, transform, val_size=0.125):
    train_data, val_data = train_test_split(train_df, test_size=val_size, random_state=42)
    
    train_dataset = VQADataset(train_data, tokenizer, transform, train_img_path, is_test=False)
    val_dataset = VQADataset(val_data, tokenizer, transform, train_img_path, is_test=False)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    return train_loader, val_loader

def train_one_epoch(model, train_loader, optimizer, criterion, device):
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

def val_one_epoch(model, val_loader, criterion, device):
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

def train(model, train_loader, val_loader, optimizer, criterion, device, epochs=1, checkpoint_path='best_model.pth'):
    best_val_loss = float('inf')
    for epoch in range(epochs):
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        avg_val_loss = val_one_epoch(model, val_loader, criterion, device)
        print(f"Epoch: {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Checkpoint saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), './data/checkpoints/resnetgpt2.pth' )
            print(f"New best model saved with val loss: {best_val_loss:.4f}")

# 데이터 불러오기
train_df = pd.read_csv('./data/train2.csv')
train_img_path = './data/image/train'

# dataset & dataloader
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
vocab_size = len(tokenizer)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 준비된 데이터로더
train_loader, val_loader = prepare_data(train_df, train_img_path, tokenizer, transform)

# Model
model = VQAModel(vocab_size).to(device)

# Criterion and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-3)

if __name__ == "__main__":
    train(model, train_loader, val_loader, optimizer, criterion, device, epochs=5)


