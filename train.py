# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:27:06 2019

@author: 12134
"""

import torchvision
import torch
from torchvision import transforms
from torch import nn
import torch.optim as optim
from autoencoder import Autoencoder
from torch.autograd import Variable
import os

zsize = 48
batch_size = 11
iterations =  500
learningRate= 0.0001

transform = transforms.Compose(
	[
	transforms.Scale((224,224), interpolation=2),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
	#transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
	])

trainset=torchvision.datasets.ImageFolder("/home/acm_mm_competition2019/dataset/images", transform=transform, target_transform=None)
trainloader = torch.utils.data.DataLoader(trainset, shuffle = True , batch_size = batch_size , num_workers = 2)

#testset=torchvision.datasets.ImageFolder("/home/deepkliv/Desktop/AE/ram/AE_classifier/fashion/dataset/test", transform=transform, target_transform=None)
#testloader = torch.utils.data.DataLoader(testset, shuffle = True , batch_size = batch_size , num_workers = 2)

autoencoder_criterion = nn.MSELoss()

autoencoder = Autoencoder()

autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr = learningRate)
list_a_loss = []

for epoch in range(iterations):
	run_loss = 0 
	autoencoder.train(True) # For training

	for i,data in enumerate(trainloader):

		inputs, labels = data
		inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

		
		autoencoder_optimizer.zero_grad()

		pred = autoencoder(inputs)
		a_loss = autoencoder_criterion(pred , inputs)
		a_loss.backward()
		autoencoder_optimizer.step()
	
		#encoder_optimizer.step()
		
		run_loss += a_loss.data[0]

		if (i +1) % 2 == 0:
			print('[%d, %5d] Autoencoder loss: %.3f' % (epoch + 1, i + 1 , run_loss/2))
			run_loss = 0.0


#		decoder_path = os.path.join('/home/acm_mm_competition2019/Decoder', 'decoder-%d.pkl' %(epoch+1))
#		encoder_path = os.path.join('/home/acm_mm_competition2019/Encoder/', 'encoder-%d.pkl' %(epoch+1))
#		autoencoder_path = os.path.join('/home/acm_mm_competition2019/Autoencoder/', 'autoencoder-%d.pkl' %(epoch+1))
		
#		torch.save(decoder.state_dict(), decoder_path)
#		torch.save(encoder.state_dict(), encoder_path)
#		torch.save(autoencoder.state_dict(), autoencoder_path)

		
#	if ( epoch+1 )% 1 == 0:
#		list_a_loss.append(run_loss/5000)
#        
#		total = 0
#		print('\n Testing ....')
#		autoencoder.train(False) # For training
#		classification.train(False)
#		for t_i,t_data in enumerate(testloader):
#						
#			if t_i * batch_size >1000:
#				break
#			t_inputs,t_labels = t_data
#			t_inputs = Variable(t_inputs).cuda()
#			t_labels = t_labels.cuda()
#			t_outputs = autoencoder(t_inputs)
#			c_pred = classification(t_inputs)
#			_, predicted = torch.max(c_pred.data, 1)
#			#print predicted.type() , t_labels.type()
#			total += t_labels.size(0)
#			correct += (predicted == t_labels).sum()
#			if (epoch + 1)%1 == 0:
#				print("saving image")
#				test_result_path = os.path.join('/home/deepkliv/Desktop/AE/ram/AE_classifier/fashion/Test_results/', 'batch_%d_%d'%((epoch+1)/1,t_i+1) + '.jpg')
#				image_tensor = torch.cat((t_inputs.data[0:8], t_outputs.data[0:8]), 0)
#				torchvision.utils.save_image(image_tensor, test_result_path)
#
#		print('Accuracy of the network on the 8000 test images: %d %%' % (100 * correct / total))

print('Finished Training and Testing')