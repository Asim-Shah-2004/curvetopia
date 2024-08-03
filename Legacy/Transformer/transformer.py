import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTFeatureExtractor, TrainingArguments, Trainer

# Step 1: Define the transformations and load the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(root=r'C:\Users\Asim\OneDrive\Desktop\projects\curvetopia\dataset', transform=transform)

# Step 2: Define a custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, feature_extractor):
        self.dataset = dataset
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = transforms.ToPILImage()(image)  # Convert tensor to PIL image
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        return {'pixel_values': inputs['pixel_values'].squeeze(), 'label': label}

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
custom_dataset = CustomDataset(dataset, feature_extractor)
dataloader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

# Step 3: Load the Vision Transformer model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=8)

# Step 4: Set up training arguments and trainer
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="no",  # Set to "no" to omit evaluation
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    weight_decay=0.01,  # Add weight decay for regularization
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=custom_dataset,
)

# Step 5: Train the model
trainer.train()

# Step 6: Save the trained model
trainer.save_model('./results')
