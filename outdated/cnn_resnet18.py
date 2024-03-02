# https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from io import BytesIO
import base64
import os
import json
import cv2
import numpy as np
from tqdm import tqdm

FILES_FOLDER = os.path.join('..', 'files')
RESULT_FILE = 'cover-cnn.json'

files = [os.path.join(FILES_FOLDER, f) for f in os.listdir(FILES_FOLDER) if os.path.isfile(os.path.join(FILES_FOLDER, f))]
print('files', len(files))

# Load the pretrained model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Use the model object to select the desired layer
layer = model._modules.get('avgpool')

# Set model to evaluation mode
model.eval()

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def extract(file):
    with open(file) as f:
        contents = f.read()


    # 1. Load the image with Pillow library
    image = np.array(Image.open(BytesIO(base64.b64decode(contents))))

    # if grayscale, transform into RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) != 3 else image
    img = Image.fromarray(rgb_image)

    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))

    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)

    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.reshape(o.data.size(1)))

    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)

    # 6. Run the model on our transformed image
    model(t_img)

    # 7. Detach our copy function from the layer
    h.remove()

    # 8. Return the feature vector
    return my_embedding

result = []

for file in tqdm(files):
    file_name = os.path.basename(file)

    item = {}
    item['id'] = file_name
    item['cover'] = extract(file).cpu().numpy().tolist()

    result.append(item)

json_object = json.dumps(result, indent=4)
with open(RESULT_FILE, "w") as outfile:
    outfile.write(json_object)