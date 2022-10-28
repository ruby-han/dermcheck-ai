# python libraties
import os
import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import seaborn as sns

# import imblearn
import logging
from tqdm import tqdm
from glob import glob
from PIL import Image
import ipywidgets

# pytorch libraries
import torch
from torch import optim,nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms
# from efficientnet_pytorch import EfficientNet

# sklearn libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# feature_extract is a boolean that defines if we are finetuning or feature extracting. 
# If feature_extract = False, the model is finetuned and all model parameters are updated. 
# If feature_extract = True, only the last layer parameters are updated, the others remain fixed.
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18, resnet34, resnet50, resnet101
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224


    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224


    elif model_name == "densenet":
        """ Densenet121
        """
#         model_ft = models.densenet121(pretrained=use_pretrained)
        model_ft = models.densenet201(pretrained=use_pretrained)
        print(type(model_ft))
        print(feature_extract)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299
        
    elif model_name == 'efficientnet':
        model_ft = EfficientNet.from_pretrained('efficientnet-b7',num_classes=num_classes)
        set_parameter_requires_grad(model_ft, feature_extract)

#         # Handle the primary net
#         num_ftrs = model_ft.fc.in_features
#         model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 600

    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft, input_size

def load_split_data(directory = None, csv_name = 'full_data', data_split = None, label = None, mode = 'all', dev_state = False, dev_sample = 15000, seed = 99):
    # Load
    data = pd.read_csv(f'{directory}/{csv_name}.csv', index_col = 0)
    
    # Subset to necessary data
    data = data[data[data_split].isna() == False]
    
    # create numerical code for each class
    data['label_idx'] = pd.Categorical(data[label]).codes
    
    # rename for final pass
    data.rename(columns = {data_split: 'dataset', 
                           label: 'label'}, 
                inplace = True)
    
    # sample, if necessary
    if dev_state:
        small_data = data.sample(n = dev_sample, random_state = seed)
        data = small_data
    
    # important cols
    cols = ['image_id', 'diagnosis', 'age', 'sex', 'localization', 'source',
            'severity', 'path', 'label', 'dataset', 'label_idx']
    data = data[cols]
    
    # Data splits
    train = data[data.dataset == 'train'].reset_index(drop = True)
    val = data[data.dataset == 'val'].reset_index(drop = True)
    test = data[data.dataset == 'test'].reset_index(drop = True)
    
    if mode == 'all':
        return data, train, test, val
        
    elif mode == 'no splits':
        return data
        
    elif mode == 'splits only':
        return train, test, val
    
class custom_loader(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index]).convert('RGB')
        
        y = torch.tensor(int(self.df['label_idx'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y
    
def build_loader(mode, df, transform, batch_size = 64, num_workers = 24):
    
    shuffle = None
    
    if mode == 'train':
        shuffle = True
    elif (mode == 'test') | (mode == 'val'):
        shuffle = False
        
        
    # Transform data
    set_trans = custom_loader(df, transform = transform)
    
    # Add to loader
    loader = DataLoader(set_trans, batch_size= batch_size, 
                              shuffle=shuffle, num_workers=num_workers)
    
    return loader

def plot_confusion(labels, predictions, normalize = True): 
    
    if normalize:
        norm = 'true'
        fmt = '.1%'
    else: 
        norm = None
        fmt = 'g'

    labs_unique = np.sort(labels.unique())

    c_matrix = confusion_matrix(labels, predictions, normalize = norm)
    plt.title("Confusion matrix")
    sns.heatmap(c_matrix, cmap='Blues', annot=True, 
                xticklabels=labs_unique, yticklabels=labs_unique, 
                fmt=fmt, cbar=True)
    plt.xlabel('predictions')
    plt.ylabel('true labels')
    plt.show()
    
def evaluate(model_name, model_source, model_dict, label_dict, show_cm = True, cm_normalize = True):
    
#     test_loader = model_dict['loader']['test_loader']
    
    model = model_dict['model']
    criterion = model_dict['criterion']
    optimizer = model_dict['optimizer']
    epochs = model_dict['epochs']
    model_directory = model_dict['mod_directory']
    model_name = model_dict['tuned_model_name']
    train_loader = model_dict['loader']['train_loader']
    val_loader = model_dict['loader']['val_loader']
    test_loader = model_dict['loader']['test_loader']
    
    # Load model
    if model_source == 'pt':
        model_in = torch.load(f'./model/{model_name}.pt')
    elif model_source == 'native':
        model_in = model
        
    # Evaluate    
    loss_test, acc_test, preds, labs = run('test', test_loader, model_in, criterion, optimizer, None, model_dict)
    
    # Flatten model labels and predictions
    labs = np.array(list(itertools.chain(*labs)))
    preds = np.array(list(itertools.chain(*preds)))
    
    # map id to label for confusion matrix
    labels = pd.Series(labs).map(label_dict)
    predictions = pd.Series(preds.flatten()).map(label_dict)
    
    # Add to data frame for easy ingestion
    pred_df = pd.concat([labels, 
           predictions, 
           pd.Series(labs), 
           pd.Series(preds.flatten())], axis = 1)\
           .rename(columns = {0:'lab', 1: 'pred', 2: 'lab_idx', 3: 'pred_idx'})
    
    if show_cm:
        plot_confusion(labels, predictions, cm_normalize)
    
    return pred_df

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def run(mode, loader, model, criterion, optimizer, epoch, model_dict):
    
    device = model_dict['device']
    total_loss, total_acc = [],[]
    mode_loss = AverageMeter()
    acc = AverageMeter() 
    true_labels = []
    predictions_out = []
    
    
    if mode == 'train':
        model.train()
        curr_iter = (epoch - 1) * len(loader)

        for i, data in enumerate(loader):
            images, labels = data
            N = images.size(0)
            # print('image shape:',images.size(0), 'label shape',labels.size(0))
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            prediction = outputs.max(1, keepdim=True)[1]
            acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)
            mode_loss.update(loss.item())
            curr_iter += 1
            if (i + 1) % 100 == 0:
    #         if (i + 1) % 1 == 0:
                print(f'[epoch {epoch}], [iter {i+1} of {len(loader)}],[{mode} loss {mode_loss.avg:.5f}], [{mode} acc {acc.avg:.5f}]')
                total_loss.append(mode_loss.avg)
                total_acc.append(acc.avg)

    elif (mode == 'test') | (mode == 'val'):
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                images, labels = data
                N = images.size(0)
                images = Variable(images).to(device)
                labels = Variable(labels).to(device)

                outputs = model(images)
                prediction = outputs.max(1, keepdim=True)[1]
                
                if mode == 'test':
                    predictions_out.append(prediction.cpu().numpy())
                    true_labels.append(labels.cpu().numpy())
                    epoch = 'test'
                
                acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)

                mode_loss.update(criterion(outputs, labels).item())

        print('------------------------------------------------------------')
        print(f'[epoch {epoch}], [{mode} loss {mode_loss.avg:.5f}], [{mode} acc {acc.avg:.5f}]')
        print('------------------------------------------------------------')
    
    if (mode == 'train') | (mode == 'val'):
        return mode_loss.avg, acc.avg
    elif mode == 'test':
        return mode_loss.avg, acc.avg, predictions_out, true_labels

def train_model(model_dict):    

    # Instantiate placeholders
    best_val_acc = 0
    total_loss_val, total_acc_val = [],[]
    total_since = time.time()
    
    # Instantiate variables from dictionary
#     train_loader = loaders['train_loader']
#     val_loader = loaders['val_loader']
    model = model_dict['model']
    criterion = model_dict['criterion']
    optimizer = model_dict['optimizer']
    epochs = model_dict['epochs']
    model_directory = model_dict['mod_directory']
    model_name = model_dict['tuned_model_name']
    train_loader = model_dict['loader']['train_loader']
    val_loader = model_dict['loader']['val_loader']
    
    
    print(f'Starting Training {model_name}')

    # Train/val loop
    for epoch in range(1, epochs+1):

        # timing
        since = time.time()
        
        # Calculate loss and acc
        loss_train, acc_train = run('train', train_loader, model, criterion, optimizer, epoch, model_dict)
        loss_val, acc_val = run('val', val_loader, model, criterion, optimizer, epoch, model_dict)

        # Track loss and acc
        total_loss_val.append(loss_val)
        total_acc_val.append(acc_val)

        # Update model state if improved
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            torch.save(model, f'{model_directory}/{model_name}.pt')

        time_elapsed = time.time() - since

        print('\nEPOCH', epoch, ":")
        print('*****************************************************')
        print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print(f'best record: [epoch {epoch}], [val loss {loss_val:.5f}], [val acc {acc_val:.5f}]')
        print('*****************************************************')

    total_time_elapsed = time.time() - total_since
    print('\nTotal run Complete in {:.0f}m {:.0f}s'.format(total_time_elapsed // 60, total_time_elapsed % 60))