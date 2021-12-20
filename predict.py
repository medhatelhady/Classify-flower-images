import argparse
from PIL import Image
import numpy as np
import json
from torchvision import transforms, models
import torch
from torch import nn, optim

def process_image(image):
    
    '''
    load image and Scales, crops, and normalizes a PIL image for a PyTorch model
    
    Args:
        image(str): store path of image
    
    Returns
        returns an Numpy array represent the image
    '''
    # load the image 
    image = Image.open(image)
    
    # create transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    # apply transform on the image 
    image = transform(image)
        
    return image

def predict(image_path, model, device, topk=5):   
    '''
    predict name of image using trained model 
    
    Args:
    image_path (str): path of image
    model (torchvision.models.densenet.DenseNet):trained model to predict name of image
    topk(int): count of higest predict names 
    
    Return
    tensor: higest classes
    tensor: probabilites of classes
    
    '''
    model.to(device)
    model.eval()
    img = process_image(image_path)
    img = img.numpy()
    img = torch.from_numpy(np.array([img])).float()

    with torch.no_grad():
        output = model.forward(img.cuda())
        
    acc = torch.exp(output).data
    
    top_k, top_class = acc.topk(topk)
    
    return top_k, top_class




parser = argparse.ArgumentParser(description='A test program.')
parser.add_argument("data_dir", help="Prints the supplied argument.",default ="flowers/train")
parser.add_argument('checkpoint', default = 'checkpoint.pth')
parser.add_argument("--top_k", default = 1, type = int)
parser.add_argument("--category_names", default="None")
parser.add_argument("--gpu", default = 'cuda')
args = parser.parse_args()

image_dir = args.data_dir

filepath = args.checkpoint

topk = args.top_k

cat_to_class = args.category_names

device = args.gpu

if cat_to_class != "None":
    
    with open(cat_to_class, 'r') as f:
        
        cat_to_class = json.load(f)
   
state = torch.load(filepath)
if state['arch'] == 'vgg16':
    saved_model = models.vgg16(pretrained=True)
    
else:
    saved_model = models.densenet121(pretrained=True)

saved_model.classifier = state['classifier']
saved_model.load_state_dict(state['state_dict'])

criterion = nn.NLLLoss()
optimizer = optim.Adam(saved_model.classifier.parameters(), lr=0.001)

saved_model.to(device)

class_to_idx = state['class_to_idx']
criterion.load_state_dict(state['criterion'])
optimizer.load_state_dict(state['optimizer'])



top_k, top_class = predict(image_dir, saved_model, device, topk)

   
top_k = top_k[0].cpu().numpy()
top_class = top_class[0].cpu().numpy()

if cat_to_class == 'None':
    for i in range(topk):
    
        print("{} : {}".format(top_class[i], top_k[i]))
else:
    idx_to_class = {val: key for key, val in class_to_idx.items()}
    categories = [idx_to_class[idx] for idx in top_class]
    labels = [cat_to_name[cat] for cat in categories]
    for i in range(topk):
        
        print("{} : {:.4f}".format(labels[i], top_k[i]))
        
          
