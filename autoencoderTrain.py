
import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
import math
#from lib.nn import SynchronizedBatchNorm2d

from torchvision import transforms
import torch.optim as optim
import numpy as np
import glob
from torch.utils import data
from PIL import Image
from tqdm import tqdm
import logging
from torch.autograd import Function
import torch.nn.functional as F

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

#cuda = torch.device('cuda:1')


model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}


zsize = 2048
batch_size = 16
iterations =  2000
learningRate= 0.0001

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
	
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

###############################################################



###############################################################
class Encoder(nn.Module):

    def __init__(self, block, layers, num_classes=23):
        self.inplanes = 64
        super (Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#, return_indices = True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, 1000)
	#self.fc = nn.Linear(num_classes,16) 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
	
        x = self.bn1(x)
        x = self.relu(x)
	
        x = self.maxpool(x)
	
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)  

encoder = Encoder(Bottleneck, [3, 4, 6, 3])
encoder.load_state_dict(load_url(model_urls['resnet50']), strict=False)
encoder.fc = nn.Linear(2048, zsize)

encoder=encoder.cuda()


##########################################################################
class Binary(Function):

    @staticmethod
    def forward(ctx, input):
        return F.relu(Variable(input.sign())).data

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

binary = Binary()
##########################################################################

class Decoder(nn.Module):
        def __init__(self):
                super(Decoder,self).__init__()
                self.dfc3 = nn.Linear(zsize, 4096)
                self.bn3 = nn.BatchNorm2d(4096)
                self.dfc2 = nn.Linear(4096, 4096)
                self.bn2 = nn.BatchNorm2d(4096)
                self.dfc1 = nn.Linear(4096,256 * 6 * 6)
                self.bn1 = nn.BatchNorm2d(256*6*6)
                self.upsample1=nn.Upsample(scale_factor=2)
                self.dconv5 = nn.ConvTranspose2d(256, 256, 3, padding = 0)
                self.dconv4 = nn.ConvTranspose2d(256, 384, 3, padding = 1)
                self.dconv3 = nn.ConvTranspose2d(384, 192, 3, padding = 1)
                self.dconv2 = nn.ConvTranspose2d(192, 64, 5, padding = 2)
                self.dconv1 = nn.ConvTranspose2d(64, 3, 12, stride = 4, padding = 4)

        def forward(self,x):#,i1,i2,i3):

                x = self.dfc3(x)
                x = F.relu(x)
               # x = F.relu(self.bn3(x))

                x = self.dfc2(x)
               # x = F.relu(self.bn2(x))
                x = F.relu(x)
                x = self.dfc1(x)
               # x = F.relu(self.bn1(x))
                x = F.relu(x)
                #print(x.size())
                x = x.view(x.shape[0],256,6,6)
                #print (x.size())
                x=self.upsample1(x)
                #print x.size()
                x = self.dconv5(x)
                #print x.size()
                x = F.relu(x)
                #print x.size()
                x = F.relu(self.dconv4(x))
                #print x.size()
                x = F.relu(self.dconv3(x))
                #print x.size()         
                x=self.upsample1(x)
                #print x.size()         
                x = self.dconv2(x)
                #print x.size()         
                x = F.relu(x)
                x=self.upsample1(x)
                #print x.size()
                x = self.dconv1(x)
                #print x.size()
                x = F.sigmoid(x)
                #print x
                return x
decoder = Decoder().cuda()
##########################################
class Autoencoder(nn.Module):
        def __init__(self):
                super(Autoencoder,self).__init__()
                self.encoder = encoder
                self.binary = Binary()
                self.decoder = Decoder().cuda()

        def forward(self,x):
                #x=Encoder(x)
                x = self.encoder(x)
                x = binary.apply(x)
                #print x
                #x,i2,i1 = self.binary(x)
                #x=Variable(x)
                x = self.decoder(x)
                return x

#print Autoencoder()

autoencoder = Autoencoder()
if torch.cuda.is_available():
        autoencoder.cuda()
        decoder.cuda()
        encoder.cuda()
print("Done")


#####################################################################################################

class Dataset(data.Dataset):
    def __init__(self, datapath):
        types = ('*.jpg','*.jpeg','*.JPG')
        self.all_data=[]
        for files in types:
            self.all_data.extend(glob.glob(datapath + files))


    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        trans = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
#        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img = Image.open(self.all_data[index])
        img = img.convert('RGB')
        img = trans(img)
        img = img.type(torch.FloatTensor)

        return (img,img)
#z = torch.zeros([16,3,224,224])   
#z = Variable(z).cuda() 
data_path = '/data/projects/acm_mm_competition2019/dataset/images/'
snapshot = '/data/projects/acm_mm_competition2019/AutoencoderFinal/autoenc2048Features-15'
#data_path = '/home/ubuntu/acm_mm_competition2019/dataset/images/'
#snapshot = '/home/ubuntu/meghana/IMSC-ACM/Autoencoder10kFinal/autoenc2048Features-15'

training_dataset = Dataset(data_path)
trainloader = data.DataLoader(training_dataset, batch_size = batch_size, num_workers = 8)

autoencoder_criterion = nn.MSELoss()

#autoencoder = Autoencoder()
#autoencoder.load_state_dict(torch.load(snapshot))
#utils.set_logger('home/ubuntu/meghana/IMSC-ACM/train.log')
logging.basicConfig(filename='training_final_2048Features.log', filemode='w',format='%(asctime)s - %(message)s',datefmt='%d-%b-%y %H:%M:%S',level=logging.INFO)
autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr = learningRate)
list_a_loss = []

logging.info("Starting training for {} epoch(s)".format(iterations))

for epoch in range(15,iterations):
    epoch_losses = []
    run_loss=0
    max_steps = 2
    train_iterator = tqdm(trainloader, total=max_steps // batch_size + 1)
    autoencoder.train(True) # For training

    for i,j in train_iterator:

        inputs, labels = i,j
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        autoencoder_optimizer.zero_grad()

        pred = autoencoder(inputs)

        a_loss = autoencoder_criterion(pred , inputs)
        epoch_losses.append(a_loss.item())
        status = 'Epoch [{0}] loss = {1:0.5f} avg = {2:0.5f}'.format(epoch + 1, a_loss.item(), np.mean(epoch_losses))

        train_iterator.set_description(status)
        logging.info(status)
##       epoch_losses.append(a_loss.item())
        a_loss.backward()
        autoencoder_optimizer.step()





#        if (i +1) % 2 == 0:
#            print('[%d, %5d] Autoencoder loss: %.3f' % (epoch + 1, i + 1 , run_loss/2))
#            run_loss = 0.0


#    decoder_path = os.path.join('/home/ubuntu/meghana/IMSC-ACM/Decoder', 'decoder-%d.pkl' %(epoch+1))
#    encoder_path = os.path.join('/home/ubuntu/meghana/IMSC-ACM/Encoder/', 'encoder-%d.pkl' %(epoch+1))
    if (epoch+1)%5 == 0:
        autoencoder_path = os.path.join('/data/projects/acm_mm_competition2019/code/AutoencoderFinal/', 'autoenc2048Features-%d' %(epoch+1))
        torch.save(autoencoder.state_dict(), autoencoder_path)


#    if ( epoch+1 )% 1 == 0:
#        list_a_loss.append(run_loss/5000)
#        
#        total = 0
#        print('\n Testing ....')
#        autoencoder.train(False) # For training
#        classification.train(False)
#        for t_i,t_data in enumerate(testloader):
#                        
#            if t_i * batch_size >1000:
#                break
#            t_inputs,t_labels = t_data
#            t_inputs = Variable(t_inputs).cuda()
#            t_labels = t_labels.cuda()
#            t_outputs = autoencoder(t_inputs)
#            c_pred = classification(t_inputs)
#            _, predicted = torch.max(c_pred.data, 1)
#            #print predicted.type() , t_labels.type()
#            total += t_labels.size(0)
#            correct += (predicted == t_labels).sum()
#            if (epoch + 1)%1 == 0:
#                print("saving image")
#                test_result_path = os.path.join('/home/deepkliv/Desktop/AE/ram/AE_classifier/fashion/Test_results/', 'batch_%d_%d'%((epoch+1)/1,t_i+1) + '.jpg')
#                image_tensor = torch.cat((t_inputs.data[0:8], t_outputs.data[0:8]), 0)
#                torchvision.utils.save_image(image_tensor, test_result_path)
#
#        print('Accuracy of the network on the 8000 test images: %d %%' % (100 * correct / total))

print('Finished Training and Testing')
