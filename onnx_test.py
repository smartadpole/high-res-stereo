import onnx

from onnxmodel import ONNXModel
from PIL import Image
import numpy as np
# from test_image import WriteDepthOnnx
from torchvision import transforms

net = ONNXModel("hsm.onnx")

limg_ori = Image.open("image/left.png").convert('RGB')
rimg_ori = Image.open("image/right.png").convert('RGB')

limg_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((384, 1280)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])(limg_ori)
rimg_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((384, 1280)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])(rimg_ori)
limg_tensor = limg_tensor.unsqueeze(0).cuda()
rimg_tensor = rimg_tensor.unsqueeze(0).cuda()

limg=limg_tensor.cpu().numpy()
rimg=rimg_tensor.cpu().numpy()

output  = net.forward(limg,rimg)
dis_array = output[1]
dis_array = (dis_array - dis_array.min()) / (dis_array.max() - dis_array.min()) * 255.0
dis_array = dis_array.astype("uint8")

import cv2
showImg = cv2.resize(dis_array, (dis_array.shape[-1], dis_array.shape[0]))
showImg = cv2.applyColorMap(cv2.convertScaleAbs(showImg, 1), cv2.COLORMAP_PARULA)
cv2.imwrite("onnx_result.jpg", showImg)
