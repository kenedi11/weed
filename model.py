import torchvision
import torch
import cv2 as cv
from torch.utils.data import Dataset,DataLoader,random_split
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import os



image_dir = r'C:\Users\Дания\colorDetection\weed\agri_data\data\train\images'
annotation_dir = r'C:\Users\Дания\colorDetection\weed\agri_data\data\train\labels'

images = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
labels = [os.path.join(annotation_dir, annot) for annot in os.listdir(annotation_dir)]

test_image_dir = r'C:\Users\Дания\colorDetection\weed\agri_data\data\test\images'
test_annotation_dir = r'C:\Users\Дания\colorDetection\weed\agri_data\data\test\labels'

test_images = [os.path.join(test_image_dir, img) for img in os.listdir(image_dir)]
test_labels = [os.path.join(test_annotation_dir, annot) for annot in os.listdir(annotation_dir)]

transformer = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize((256, 256))])


class WeedData(Dataset):
  def __init__(self, images, labels, transformer):
    self.images = images
    self.labels = labels
    self.transformer = transformer
  def __getitem__(self, item):
        img = Image.open(self.images[item]).convert('RGB')
        img = transformer(img)
        df = pd.read_csv(self.labels[item], header=None, sep=' ')
        df.columns = ['label', 'x_cen', 'y_cen', 'w', 'h']
        df['xmin'] = (df['x_cen'] - df['w'] / 2) * 512
        df['ymin'] = (df['y_cen'] - df['h'] / 2) * 512
        df['xmax'] = (df['x_cen'] + df['w'] / 2) * 512
        df['ymax'] = (df['y_cen'] + df['h'] / 2) * 512
        bbox = np.array(df.iloc[:,5:]).tolist()
        label = np.array(df.iloc[:, 0]).squeeze().tolist()
        bbox = torch.tensor(bbox, dtype=torch.float32)
        label = torch.as_tensor(label, dtype=torch.int64).reshape(-1,)
        target = {}
        target['boxes'] = bbox
        target['labels'] = label
        return img, target

  def __len__(self):
        return len(self.images)

train = WeedData(images, labels, transformer)
test = WeedData(test_images, test_labels, transformer)

def detection_collate(x):
    return list(tuple(zip(*x)))


train_dl = DataLoader(dataset = train, batch_size = 32, shuffle = True, collate_fn = detection_collate, pin_memory = True)
test_dl = DataLoader(dataset = test, batch_size = 32, shuffle = True, collate_fn = detection_collate, pin_memory = True)
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

in_features = model.roi_heads.box_predictor.cls_score.in_features
n_classes = 2
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.16, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    i = 0
    for images, targets in tqdm(train_dl):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        i += 1
        if i % 10 == 0:
            print(f"Iteration {i}, Loss: {losses.item()}")

    lr_scheduler.step()

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for images, targets in test_dl:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
        
        val_loss /= len(test_dl)
        print(f"Validation Loss after epoch {epoch + 1}: {val_loss:.4f}")
    
    model.train()

    # Save the model after each epoch
    model_path = f'models/fasterrcnn_resnet50_fpn_epoch{epoch + 1}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")