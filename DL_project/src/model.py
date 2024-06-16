import torch
import torch.nn as nn
import torchvision.models as models
from transformers import GPT2Model

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, features, mask=None):
        # Compute attention scores
        attn_scores = self.attn(features)  # Shape: (batch_size, seq_len, 1)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)  # Shape: (batch_size, seq_len, 1)

        # Compute weighted sum of features
        weighted_features = torch.sum(attn_weights * features, dim=1)  # Shape: (batch_size, hidden_dim)
        return weighted_features

class VQAModel(nn.Module):
    def __init__(self, vocab_size):
        super(VQAModel, self).__init__()
        self.vocab_size = vocab_size

        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 512)
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.gpt2.resize_token_embeddings(vocab_size)  # 추가한 [PAD] 토큰 반영

        self.att_pool = AttentionPooling(self.gpt2.config.hidden_size)
        combined_features_size = 512 + self.gpt2.config.hidden_size  # resnet 출력 차원 + gpt2 출력 차원
        self.classifier = nn.Linear(combined_features_size, 3129)

    def forward(self, images, question, padding_mask):
        # Extract features from the image using ResNet
        img_features = self.resnet(images)  # Shape: (batch_size, 512)
        
        # Extract features from the text using GPT-2
        text_features = self.gpt2(input_ids=question, attention_mask=padding_mask).last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)
        
        # Apply attention pooling to the text features
        text_features_pooled = self.att_pool(text_features, padding_mask)  # Shape: (batch_size, hidden_dim)

        # Concatenate image features and pooled text features
        combined_features = torch.cat((img_features, text_features_pooled), dim=1)  # Shape: (batch_size, 512 + hidden_dim)
        
        # Pass the combined features through the fully connected layer
        outputs = self.classifier(combined_features)  # Shape: (batch_size, 3129)
        return outputs









# import torch.nn as nn
# import torch
# from transformers import DistilBertModel
# from torchvision import models
# import torch.nn.functional as F

# class CompactBilinearPooling(nn.Module):
#     def __init__(self, input_dim1, input_dim2, output_dim=1024):
#         super(CompactBilinearPooling, self).__init__()
#         self.output_dim = output_dim
#         self.S1 = nn.Parameter(torch.randint(0, 2, (input_dim1, output_dim)) * 2 - 1, requires_grad=False)
#         self.S2 = nn.Parameter(torch.randint(0, 2, (input_dim2, output_dim)) * 2 - 1, requires_grad=False)
#         self.H1 = nn.Parameter(torch.randn(input_dim1, output_dim), requires_grad=False)
#         self.H2 = nn.Parameter(torch.randn(input_dim2, output_dim), requires_grad=False)

#     def forward(self, input1, input2):
#         sketch1 = torch.einsum('bij,jk->bik', input1, self.S1 * self.H1)
#         sketch2 = torch.einsum('bij,jk->bik', input2, self.S2 * self.H2)
#         result = torch.fft.irfft(sketch1 * sketch2, n=self.output_dim, dim=-1)
#         return result

# class VQAModel(nn.Module):
#     def __init__(self, vocab_size):
#         super(VQAModel, self).__init__()
#         self.vocab_size = vocab_size

#         # Load pre-trained resnet50 without the last fc layer
#         self.resnet = models.resnet50(pretrained=True)
#         self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # remove last fc layer and avgpool layer

#         self.distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
#         self.distilbert_model.resize_token_embeddings(vocab_size)

#         self.image_transform = nn.Linear(2048, 1024)  # Update the input dimension to 2048 (resnet50 last feature map depth)
#         self.question_transform = nn.Linear(768, 1024)  # Add this layer to match the dimension to 1024

#         self.mcb = CompactBilinearPooling(1024, 1024, 1024)  # Ensure input_dim1 matches the output of image_transform
#         self.conv1D = nn.Conv1d(1024, 1024, kernel_size=1)  # 1D convolution

#         self.classifier = nn.Linear(1024, 3129)  # Ensure classifier output matches the number of classes

#     def forward(self, images, question, padding_mask):
#         image_features = self.resnet(images)  # Now, this will give feature maps
#         image_features_reshaped = image_features.view(image_features.size(0), image_features.size(1), -1).mean(2)
#         image_features_transformed = self.image_transform(image_features_reshaped).unsqueeze(1)

#         outputs = self.distilbert_model(question, attention_mask=padding_mask)
#         question_features = outputs.last_hidden_state.mean(dim=1)
#         question_features_transformed = self.question_transform(question_features).unsqueeze(1)

#         mcb1_output = self.mcb(image_features_transformed, question_features_transformed)

#         conv_output = self.conv1D(mcb1_output.transpose(1, 2))

#         attention_weights = F.softmax(conv_output, dim=-1)

#         weighted_image_features_sum = (attention_weights * image_features_transformed.transpose(1, 2)).sum(dim=-1, keepdim=True)
#         weighted_image_features_sum = weighted_image_features_sum.transpose(1, 2)

#         mcb2_output = self.mcb(weighted_image_features_sum, question_features_transformed)

#         output = self.classifier(mcb2_output.squeeze(1))
#         return output

# import torch
# import torch.nn as nn
# import torchvision.models as models
# from transformers import GPT2Model

# class CompactBilinearPooling(nn.Module):
#     def __init__(self, input_dim1, input_dim2, output_dim):
#         super(CompactBilinearPooling, self).__init__()
#         self.output_dim = output_dim

#         # Random Fourier embeddings
#         self.sketch_matrix1 = nn.Parameter(torch.randn(input_dim1, output_dim))
#         self.sketch_matrix2 = nn.Parameter(torch.randn(input_dim2, output_dim))

#     def forward(self, x, y):
#         batch_size, _, _ = x.size()
#         x_flat = x.reshape(-1, x.size(-1))
#         y_flat = y.reshape(-1, y.size(-1))

#         # Sketch embeddings
#         x_sketch = torch.matmul(x_flat, self.sketch_matrix1)
#         y_sketch = torch.matmul(y_flat, self.sketch_matrix2)

#         # Element-wise product
#         xy = x_sketch * y_sketch

#         # Reshape back to batch size
#         return xy.reshape(batch_size, -1, self.output_dim)

# class VQAModel(nn.Module):
#     def __init__(self, vocab_size, num_classes=3129, output_dim=8192):
#         super(VQAModel, self).__init__()
#         self.vocab_size = vocab_size
#         self.num_classes = num_classes
#         self.output_dim = output_dim

#         self.resnet = models.resnet50(pretrained=True)
#         self.gpt2 = GPT2Model.from_pretrained('gpt2')
#         self.gpt2.resize_token_embeddings(vocab_size)  # Reflect added [PAD] token

#         self.pooling = CompactBilinearPooling(1000, self.gpt2.config.hidden_size, output_dim)
#         self.classifier = nn.Linear(output_dim, num_classes)
#         self._initialize_weights()

#     def _initialize_weights(self):
#         nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out', nonlinearity='relu')
#         if self.classifier.bias is not None:
#             nn.init.constant_(self.classifier.bias, 0)

#     def forward(self, images, question, padding_mask=None):
#         image_features = self.resnet(images)
#         if check_for_nan_inf(image_features, "image_features after resnet"):
#             return torch.tensor([float('nan')], requires_grad=True)

#         image_features = image_features.view(image_features.size(0), -1)
#         if check_for_nan_inf(image_features, "image_features after view"):
#             return torch.tensor([float('nan')], requires_grad=True)

#         outputs = self.gpt2(question, attention_mask=padding_mask)
#         output_features = outputs.last_hidden_state  # [batch, sequence, hidden]
#         if check_for_nan_inf(output_features, "output_features after gpt2"):
#             return torch.tensor([float('nan')], requires_grad=True)

#         image_features = image_features.unsqueeze(1).expand(-1, output_features.size(1), -1)  # [batch, sequence, 1000]
#         if check_for_nan_inf(image_features, "image_features after unsqueeze and expand"):
#             return torch.tensor([float('nan')], requires_grad=True)

#         combined = self.pooling(image_features, output_features)  # [batch, sequence, output_dim]
#         if check_for_nan_inf(combined, "combined after pooling"):
#             return torch.tensor([float('nan')], requires_grad=True)

#         if padding_mask is not None:
#             combined = combined * padding_mask.unsqueeze(-1).float()  # Mask padding tokens
#             if check_for_nan_inf(combined, "combined after masking"):
#                 return torch.tensor([float('nan')], requires_grad=True)

#         combined = combined.mean(dim=1)  # Pool over the sequence dimension
#         if check_for_nan_inf(combined, "combined after mean pooling"):
#             return torch.tensor([float('nan')], requires_grad=True)

#         output = self.classifier(combined)  # [batch, num_classes]
#         output = torch.clamp(output, min=-1e9, max=1e9)  # Clamp outputs to avoid extreme values
#         if check_for_nan_inf(output, "output after classifier and clamping"):
#             return torch.tensor([float('nan')], requires_grad=True)

#         return output

# def check_for_nan_inf(tensor, name):
#     if torch.isnan(tensor).any():
#         print(f"{name} contains NaN values.")
#         return True
#     if torch.isinf(tensor).any():
#         print(f"{name} contains Inf values.")
#         return True
#     return False

# def check_data_for_nan_inf(data):
#     for key, value in data.items():
#         if check_for_nan_inf(value, key):
#             return True
#     return False





        




