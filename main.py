import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from Caltech256 import Caltech256



from torchvision import transforms #add this line in the above snippet
from torch.utils.data import DataLoader #add this line in the above snippet
matplotlib.use('Agg')
plt.ion()   # interactive mode


example_transform = transforms.Compose(
    [
        transforms.Scale((224,224)),
        transforms.ToTensor(),
    ]
)
        
caltech256_train = Caltech256("/datasets/Caltech256/256_ObjectCategories/", example_transform, train=True)
caltech256_test = Caltech256("/datasets/Caltech256/256_ObjectCategories/", example_transform, train=False)

train_data = DataLoader(
    dataset = caltech256_train,
    batch_size = 32,
    shuffle = True,
    num_workers = 4
)

test_data = DataLoader(
    dataset = caltech256_test,
    batch_size = 32,
    shuffle = True,
    num_workers = 4
)

use_gpu = torch.cuda.is_available()
print(use_gpu)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
	
	    count = 0
            # Iterate over data.
            for data in train_data:
                # get the inputs
                inputs, labels = data
		count += labels.size(0)
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.long().cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels.long())
		
		labels = labels - 1		
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels.squeeze(1))
		running_loss += loss.data[0]
                loss.backward()
		optimizer.step()

	    print('Loss of the network on the train data: %.3f' % (
    	          running_loss))	

	    model.eval()	
	    correct = 0
	    total = 0
	    for data in train_data:
    	        images, labels = data
   	        outputs = model(Variable(images.cuda()))
		labels = labels - 1

   	        _, predicted = torch.max(outputs.data, 1)
   	        total += labels.size(0)
    	        correct += (predicted == labels.long().cuda()).sum()

	    print('Accuracy of the network on the train data: %d %%' % (
    	         100 * correct / total))	

	    correct = 0
	    total = 0
	    loss_ = 0.0
	    for data in test_data:
                  
    	        images, labels = data
                labels = labels.long().cuda()
   	        outputs = model(Variable(images.cuda()))
		labels = labels - 1

                loss = criterion(outputs, Variable(labels).squeeze(1))
		loss_ += loss.data[0]

   	        _, predicted = torch.max(outputs.data, 1)
   	        total += labels.size(0)
    	        correct += (predicted == labels).sum()

	    print('Accuracy of the network on the test data: %d %%' % (
    	         100 * correct / total))	
	    print('Loss of the network on the test data: %.3f' % (
    	          loss_))	
    return model

# Finetune
model_ft = models.vgg16(num_classes=1000, pretrained='imagenet')
for param in model_ft.parameters():
	param.requires_grad = False

model_ft.classifier._modules['6'] = nn.Linear(4096, 256)
print(model_ft)

if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.classifier._modules['6'].parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=20)

vis_data = DataLoader(
    dataset = caltech256_train,
    batch_size = 1,
    shuffle = True,
    num_workers = 4
)

def imshow(img,name):
    """Imshow for Tensor."""
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg), interpolation='nearest')
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.savefig(name)
	
# Get a batch of training data
vis_inputs, vis_classes = next(iter(vis_data))
# Make a grid from batch
out = torchvision.utils.make_grid(vis_inputs)
imshow(out, 'foo.png')
model_ft = model_ft.features
model_ft.eval()
x = Variable(vis_inputs.cuda())
'''
outputs = model_ft(x)	

out = torchvision.utils.make_grid(mo.parameters())
imshow(out, 'conv1.png')
'''

for index, layer in enumerate(model_ft):
	x = layer(x)
	if index == 1:
		out = torchvision.utils.make_grid(x.data[0])
		print(out[0])
		imshow(out[0], 'conv1.png')
	elif index == 29:
		out = torchvision.utils.make_grid(x.data[0])
		imshow(out[0], 'conv2.png')
	
