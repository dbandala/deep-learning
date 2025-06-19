
# git clone https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch.git
# cd Yet-Another-EfficientDet-Pytorch
# pip install -r requirements.txt


import torch
from efficientdet.model import EfficientDet
from efficientdet.utils import preprocess, postprocess
import cv2
import matplotlib.pyplot as plt

# Load the pre-trained EfficientDet model
model = EfficientDet(compound_coef=0, num_classes=90)
model.load_state_dict(torch.load('efficientdet-d0.pth'))
model.eval()

# Load and preprocess the image
image_path = 'path_to_your_image.jpg'
image = cv2.imread(image_path)
input_tensor = preprocess(image)

# Perform inference
with torch.no_grad():
    outputs = model(input_tensor)

# Postprocess and visualize the results
boxes, scores, labels = postprocess(outputs, threshold=0.5)
for box, score, label in zip(boxes, scores, labels):
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label_text = f'Class {label}: {score:.2f}'
    cv2.putText(image, label_text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
