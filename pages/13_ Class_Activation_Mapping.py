######
# This is a copy of official CAM-repo
# https://github.com/zhoubolei/CAM
######

# simple implementation of CAM in PyTorch for the modelworks such as Resmodel, Densemodel, Squeezemodel, Inception
# last update by BZ, June 30, 2021

from PIL import Image
import torch
from torchvision import models, transforms
from torch.nn import functional as F
import numpy as np
import cv2
import json
import streamlit as st

# load the imagemodel category list

# input image
LABELS_file = "imagenet-simple-labels.json"
# image_file = "dog.jpeg"



with open(LABELS_file) as f:
    classes = json.load(f)
# modelworks such as googlemodel, resmodel, densemodel already use global average pooling at the end, so CAM could be used directly.

@st.cache_resource
def load_models():
    rn34_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    rn152_model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    dn_model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
    return rn34_model, rn152_model, dn_model

rn34_finalconv_name = "layer4"  # this is the last conv layer of the modelwork

rn152_finalconv_name = "layer4"

dn_finalconv_name = "features"


model_list = load_models()
model_names = ('rn34', 'rn152', 'dn')
final_conv_names = (rn34_finalconv_name, rn152_finalconv_name, dn_finalconv_name)

_ = [model.eval() for model in model_list]

# hook the feature extractor
features_blobs = []


def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())



for model, final_conv in zip(model_list, final_conv_names):
    model._modules.get(final_conv).register_forward_hook(hook_feature)

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    print(feature_conv.shape)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
)

columns = st.columns(3)
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file:
    img_pil = Image.open(uploaded_file).convert('RGB')
    img_tensor = preprocess(img_pil)
    

    for ix, (model, name) in enumerate(zip(model_list, model_names)):
        with torch.inference_mode():
            logit = model(img_tensor.unsqueeze(0))
        h_x = F.softmax(logit, dim=1).squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.numpy()
        idx = idx.numpy()

        # output the prediction
        for i in range(0, 5):
            print(f"{probs[i]:.3f} -> {classes[idx[i]]}")
        weight_softmax = np.squeeze(list(model.parameters())[-2].data.numpy())
        # generate class activation mapping for the top1 prediction
        CAMs = returnCAM(features_blobs[ix], weight_softmax, [idx[0]])

        # render the CAM and output
        print(f"output CAM.jpg for the top1 prediction: {classes[idx[0]]}" )
        
        cv2_image = cv2.cvtColor(np.array(Image.open(uploaded_file)), cv2.COLOR_BGR2RGB)
        # st.write(cv2_image.shape)
        height, width, _ = cv2_image.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + cv2_image * 0.5
        with columns[ix]:
            st.subheader(model_names[ix])
            st.image(Image.fromarray(result.astype(np.uint8))) 


