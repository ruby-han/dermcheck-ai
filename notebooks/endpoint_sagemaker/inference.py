# credits: https://github.com/abiodunjames/MachineLearning/blob/master/DeployYourModelToSageMaker/inference.py
# medium article: https://samuelabiodun.medium.com/how-to-deploy-a-pytorch-model-on-sagemaker-aa9a38a277b6

import json
import logging
import os
import torch
import requests
from PIL import Image
from torchvision import transforms
from torchvision import models
import torch.nn as nn

logger = logging.getLogger(__name__)


def model_fn(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading the model.')
    # model = models.resnet50(pretrained=False)
#     fc_inputs = model.fc.in_features

#     model.fc = nn.Sequential(
#         nn.Linear(fc_inputs, 2048),
#         nn.ReLU(inplace=True),
#         nn.Linear(2048, 10),
#         nn.Dropout(0.4),
#         nn.LogSoftmax(dim=1))
    model = models.resnet50(weights='DEFAULT')
    model.fc = nn.Linear(2048, 6, bias=True)

    with open(os.path.join(model_dir, 'merged_resnet50.pt'), 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()
    logger.info('Done loading model')
    return model


def input_fn(request_body, content_type='application/json'):
    logger.info('Deserializing the input data.')
    if content_type == 'application/json':
        input_data = json.loads(request_body)
        url = input_data['url']
        logger.info(f'Image url: {url}')
        image_data = Image.open(requests.get(url, stream=True).raw)

        image_transform = transforms.Compose([
            transforms.Resize(size=255),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.676, 0.542, 0.519], std=[0.290, 0.226, 0.237])
        ])

        return image_transform(image_data)
    raise Exception(f'Requested unsupported ContentType in content_type: {content_type}')


def output_fn(prediction_output, accept='application/json'):
    logger.info('Serializing the generated output.')
    classes = {0: 'Non-Cancerous Skin Condition', 1: 'Toxin, Fungal, Bug, Viral, or Bacterial Infections', 2: 'Unclassified', 3: 'Potentially Malignant Skin Tumors', 4: 'Benign Marking or Mole', 5: 'Autoimmue Disorder'}

    topk, topclass = prediction_output.topk(3, dim=1)
    result = []

    for i in range(3):
        pred = {'prediction': classes[topclass.cpu().numpy()[0][i]], 'score': f'{topk.cpu().numpy()[0][i] * 100}%'}
        logger.info(f'Adding pediction: {pred}')
        result.append(pred)

    if accept == 'application/json':
        return json.dumps(result), accept
    raise Exception(f'Requested unsupported ContentType in Accept: {accept}')


def predict_fn(input_data, model):
    logger.info('Generating prediction based on input parameters.')
    if torch.cuda.is_available():
        input_data = input_data.view(1, 3, 224, 224).cuda()
    else:
        input_data = input_data.view(1, 3, 224, 224)

    with torch.no_grad():
        model.eval()
        out = model(input_data)
        ps = torch.exp(out)

    return ps