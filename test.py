import cv2
import torch
import matplotlib.pyplot as plt

model = r'C:\Users\Дания\colorDetection\12test_mIou_0.7400000095367432test_loss0.11100000143051147 (1).pth'

names = {'0': 'crop', '1': 'weed'}
src_img = plt.imread(r"C:\Users\Дания\colorDetection\data\agri_0_113.jpeg")
img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

img_tensor = torch.from_numpy(img/255.).permute(2,0,1).float().cuda()

out = model(torch.unsqueeze(img_tensor,dim=0))
boxes = out[0]['boxes'].cpu().detach().numpy().astype(int)
labels = out[0]['labels'].cpu().detach().numpy()
scores = out[0]['scores'].cpu().detach().numpy()
for idx in range(boxes.shape[0]):
    if scores[idx] >= 0.8:
        x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
        name = names.get(str(labels[idx].item()))
        cv2.rectangle(src_img,(x1,y1),(x2,y2),(255,0,0),thickness=1)
        cv2.putText(src_img, text=name, org=(x1, y1+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))

plt.imshow(src_img)