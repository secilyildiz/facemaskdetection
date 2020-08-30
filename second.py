import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


transformations = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_set = datasets.ImageFolder("root/label/train", transform = transformations)
val_set = datasets.ImageFolder("root/label/valid", transform = transformations)


train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size =32, shuffle=True)



model = models.densenet161(pretrained=True)

for param in model.parameters():
    param.requires_grad = False


classifier_input = model.classifier.in_features
num_labels = #PUT IN THE NUMBER OF LABELS IN YOUR DATA
classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                           nn.ReLU(),
                           nn.Linear(1024, 512),
                           nn.ReLU(),
                           nn.Linear(512, num_labels),
                           nn.LogSoftmax(dim=1))

model.classifier = classifier



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters())


epochs = 10
for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    accuracy = 0
    
    model.train()
    counter = 0
    for inputs, labels in train_loader:
    
        inputs, labels = inputs.to(device), labels.to(device)
 
        optimizer.zero_grad()
     
        output = model.forward(inputs)
        
        loss = criterion(output, labels)
       
        loss.backward()
    
        optimizer.step()
        
        train_loss += loss.item()*inputs.size(0)
        
       
        counter += 1
        print(counter, "/", len(train_loader))
        

    model.eval()
    counter = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            
            inputs, labels = inputs.to(device), labels.to(device)
           
            output = model.forward(inputs)
            
            valloss = criterion(output, labels)
            
            val_loss += valloss.item()*inputs.size(0)
            
            
            output = torch.exp(output)
          
            top_p, top_class = output.topk(1, dim=1)
            
            equals = top_class == labels.view(*top_class.shape)
            
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
           
            counter += 1
            print(counter, "/", len(val_loader))
    
    
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = val_loss/len(val_loader.dataset)
 
    print('Accuracy: ', accuracy/len(val_loader))
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))




model.eval()



def process_image(image_path):
 
    img = Image.open(image_path)
    

    width, height = img.size
    
   
    img = img.resize((255, int(255*(height/width))) if width < height else (int(255*(width/height)), 255))
    
   
    width, height = img.size
    
    
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    img = img.crop((left, top, right, bottom))
    
   
    img = np.array(img)
    
    img = img.transpose((2, 0, 1))
    
   
    img = img/255
    
   
    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225
    
    
    img = img[np.newaxis,:]
    
   
    image = torch.from_numpy(img)
    image = image.float()
    return image


def predict(image, model):
  
    output = model.forward(image)
 
    output = torch.exp(output)
    
    probs, classes = output.topk(1, dim=1)
    return probs.item(), classes.item()



def show_image(image):

    image = image.numpy()
    

    image[0] = image[0] * 0.226 + 0.445
