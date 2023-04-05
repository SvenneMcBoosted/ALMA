import torch
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import torch.optim as optim
import pandas as pd
import os, subprocess, sys
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
from astropy.io import fits




class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, impath_index, 
    label_index, label_fn, usecols=[0,1], transform=None):
        self.img_labels = pd.read_csv(annotations_file, usecols=usecols,sep=' ')
        self.img_dir = img_dir
        self.impath_index = impath_index
        self.label_index = label_index
        self.label_fn = label_fn
        self.transform = transform
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, self.impath_index])
        print(img_path)
        image = load_fits_as_tensor(img_path + '.fits')
        print(image.shape)
        label = self.label_fn(self.img_labels.iloc[idx, self.label_index])
        if self.transform:
            image = self.transform(image)
        return image, label
    
    
def load_fits_as_tensor(filename):
        """Read a FITS file from disk and convert it to a Torch tensor."""
        fits_np = fits.getdata(filename, memmap=False)
        return torch.from_numpy(fits_np.astype(np.float32))

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = X.float()
        y[y > 1] = 1

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.float()
            y[y > 1] = 1
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def epoch_loop(epochs, train_dataloader, val_dataloader=None):
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, criterion, optimizer)
        if val_dataloader is not None:
            test(val_dataloader, model, criterion)
        step_lr_scheduler.step()
    print("Done!")
    
# ====================== DATASETS =======================
batch_size = 32
img_transform = transforms.Compose([transforms.Resize((128, 128))])

er_label_fn = lambda s: int(s)
kg_label_fn = lambda s: int(s == "few varrao, hive beetles" or  s == "Varroa, Small Hive Beetles")

er_train_set = CustomImageDataset(  "C:/Users/jensc/Documents/GitHub/ALMA/Code/data/train/annotations.csv", 
                                    "C:/Users/jensc/Documents/GitHub/ALMA/Code/data/train", 
                                    0, 
                                    1, 
                                    er_label_fn)

er_val_set = CustomImageDataset(    "C:/Users/jensc/Documents/GitHub/ALMA/Code/data/train/annotations.csv", 
                                    "C:/Users/jensc/Documents/GitHub/ALMA/Code/data/train", 
                                    0, 
                                    1, 
                                    er_label_fn)


er_train_loader = DataLoader(er_train_set, batch_size=batch_size, shuffle=True)
er_val_loader = DataLoader(er_val_set, batch_size=batch_size)

# ======================= MODEL =========================

should_train = True if input("Load (1) or train (0)?\n") == "0" else False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_ftrs, 2), nn.Softmax())

if not should_train:
    model.load_state_dict(torch.load("model.pth", map_location=torch.device(device)))

model.to(device)

# ===================== OPTIMIZER =======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# ===================== TRAINING ========================

if should_train:
    epoch_loop(10, er_train_loader, er_val_loader)
    torch.save(model.state_dict(), "model.pth")

# ============== COPY IMAGES TO FOLDER =================
inp = input("Use current images (1), copy images to folder (2) or exit (0)? \n")
if inp != "1" and inp != "2": exit()
gen_images = True if inp == "2" else False

classes = ["not infected", "infected"]

if gen_images:
    n_images = int(input("Number of images to copy? \n"))
    inf = int(int(input("Number of infected in %? \n")) / 100 * n_images)
    not_inf = n_images - inf

    cur_inf = 0
    cur_not_inf = 0
    print("Copying...")
    for img, label in er_val_set:
        img = img[None, :, :, :]
        img = img.float()
        img = img/255
        label = 1 if label > 0 else 0
        img_index = cur_inf + cur_not_inf
        if label == 0 and cur_not_inf < not_inf:
            print("images/img_" + str(img_index) + ".png: " + classes[label])
            save_image(img, f"images/img_" + str(img_index) + ".png")
            cur_not_inf += 1
        elif label == 1 and cur_inf < inf:
            print("images/img_" + str(img_index) + ".png: " + classes[label])
            save_image(img, f"images/img_" + str(img_index) + ".png")
            cur_inf += 1
        if img_index == n_images:
            print("Copying images complete!")
            break


# ================== PREDICTION =======================

def open_file(filename):
    image = mpimg.imread(filename)
    plt.title(filename)
    plt.imshow(image)
    plt.show(block=False)
        
model.eval()
with torch.no_grad():
    i = 0
    images = []
    n_images = int(input("Number of images to predict? \n"))
    infected = 0
    print("Predicting...")
    list = []
    unsure_num = 0
    for i in range(n_images):
        img_path = "images\\img_" + str(i) + ".fits" if sys.platform == "win32" else "images/img_" + str(i) + ".fits"
        img = load_fits_as_tensor(img_path)
        print(img.shape())
        img = img.to(device)
        img = img[None, :, :, :]
        img = img.float()
        
        pred = model(img)
        predicted = classes[pred[0].argmax(0)]
        pred_num = round(pred[0][0].item(), 3)
        check = "!" if (pred_num < 0.8 and pred_num > 0.2) else " " 
        if (pred_num < 0.8 and pred_num > 0.2):
            list.append([img_path, predicted, pred_num])
            unsure_num += 1
        #print(pred)
        print(f'images/img_{i}.png "{predicted}" {round(pred[0][0].item(), 3)} {check}')
        if pred[0].argmax(0) == 1:
            infected += 1

    print("Predicted that " + str(round(infected/n_images*100)) + "% is infected")
    if (unsure_num > 0):
        should_open = True if input(f"There are {unsure_num} predictions that need reviewing. Do you want to open the images (1), or exit (0)? \n") == "1" else False
        if should_open:
            added_inf = 0
            for img in list:
                print(f'"{img[0]}" is predicted to be "{img[1]}" with rating {img[2]}')
                file_r = open_file(img[0])
                inp = input("Do you think it is infected (1) or not (0)? Exit (e). \n") 
                inf = False
                if inp != "e":
                    inf = True if inp == "1" else False
                else:
                    exit()
                if inf and img[1] == "not infected":
                    added_inf += 1
                elif not inf and img[1] == "infected":
                    added_inf -= 1
                print()
                if sys.platform != "win32": plt.close()
        
            new_infected = infected + added_inf
            print(f"New prediction is that {round(new_infected/n_images*100)} % is infected. The old prediction was {round(infected/n_images*100)}%.")