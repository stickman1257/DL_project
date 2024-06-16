from dataclasses import dataclass
import datetime
import os
import random
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Tokenizer, DistilBertTokenizer

from transformers import get_cosine_schedule_with_warmup

from src.dataset import VQADataset
from src.model import VQAModel

import warnings
warnings.filterwarnings(action='ignore')

# Hyperparameter Setting
CFG = {
    'MODEL_SIZE': 'large',
    'IMG_SIZE': 480,
    'EPOCHS': 100,
    'LEARNING_RATE': 2e-3,
    'BATCH_SIZE': 32,
    'SEED': 41
}


@dataclass
class DistributedArgs:
    world_size: int = 5  # 총 GPU 개수
    gpu: tuple = (0, 1, 2, 3, 4)  # 각 GPU ID
    dist_url: str = "tcp://127.0.0.1:38381"  # 분산 학습을 위한 URL
    dist_backend: str = 'nccl'  # 분산 학습 백엔드


# Fixed RandomSeed
random.seed(CFG['SEED'])
os.environ['PYTHONHASHSEED'] = str(CFG['SEED'])
np.random.seed(CFG['SEED'])
torch.manual_seed(CFG['SEED'])
torch.cuda.manual_seed(CFG['SEED'])
torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_sharing_strategy('file_system')
def check_for_nan_inf(tensor, name):
    if torch.isnan(tensor).any():
        print(f"{name} contains NaN values.")
        return True
    if torch.isinf(tensor).any():
        print(f"{name} contains Inf values.")
        return True
    return False

def main():
    args = DistributedArgs()
    mp.spawn(main_worker, args=(args,), nprocs=args.world_size, join=True)


def main_worker(rank, args):
    torch.cuda.set_device(args.gpu[rank])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=rank,
        timeout=datetime.timedelta(0, 7200)
    )
    torch.distributed.barrier()

    # Data Load
    train_df = pd.read_csv('./data/train2.csv')
    train_img_path = './data/image/train'

    # dataset & dataloader
    
    # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    vocab_size = len(tokenizer)
    train_dataset = VQADataset(train_df, tokenizer, train_img_path, img_size=CFG['IMG_SIZE'], is_train=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], num_workers=4, sampler=train_sampler, pin_memory=True)

    

    model = VQAModel(vocab_size).to(device)
    model.to(device)
    model = DDP(model, device_ids=[args.gpu[rank]], find_unused_parameters=False)

    # Train
    
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=CFG["LEARNING_RATE"], betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=len(train_loader) * int(CFG["EPOCHS"] * 0.1),
        num_training_steps=len(train_loader) * CFG["EPOCHS"]
    )
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    for epoch in range(1, CFG['EPOCHS']+1):
        total_loss = 0
        stop_training = False

        for data in tqdm(train_loader, total=len(train_loader)):
            images = data['image'].to(device)
            question = data['question'].to(device)
            padding_mask = data['padding_mask'].to(device)
            answer = data['answer'].to(device)

            optimizer.zero_grad()
           
            with torch.cuda.amp.autocast():
                outputs = model(images, question, padding_mask)
                if check_for_nan_inf(outputs, "Model outputs"):
                    stop_training = True
                    break
                loss = criterion(input=outputs.float(), target=answer.float())
                if check_for_nan_inf(loss, "Loss"):
                    stop_training = True
                    break
            total_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

        if stop_training:
            print(f"Training stopped at epoch {epoch} due to NaN/Inf values.")
            break

        if rank == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch}/{CFG["EPOCHS"]}], Train Loss: [{avg_loss:.5f}]')

            torch.save(
                model.state_dict(),
                os.path.join(
                    './data/checkpoints/resgpt',
                    f'{epoch}_{CFG["EPOCHS"]}_{"{:.0e}".format(CFG["LEARNING_RATE"])}_resgpt2ver1_model.pt'
                )
            )


if __name__ == '__main__':
    main()
