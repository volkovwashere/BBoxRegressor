import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import random
import cv2
import matplotlib
from cifar10_models import mobilenetv2

#Parameters
BatchLength=1  #1 image is in a minibatch
Size=[3,32, 32] #Input img will be resized to this size
NumClasses = 10
batch_size=1




def GeneticStickerAttack(Img, Net,Pred, GTLabel, GoalClass,ImageIndex):
	OrigGt=Pred[GTLabel]
	OrigGoal=Pred[GoalClass]
	OrigPred=Pred
	MinWeight=OrigGt-OrigGoal
	MinPos=[]
	MinSize=[]
	Minindex=0
	MindPred=np.zeros(NumClasses)
	AttackedImg= np.zeros((3,32,32))
	InitialValue=(MinWeight)
	
	GenomeSize=450
	NumSteps=500
	KeepRatio=0.2
	NewRatio=0.2
	StickerNum=4
	StickerColor=[0.0, 0.0, 1.0, 1.0]
	MaxStickerSize=5
	MinStickersize=0
	GeneratedRatio=1-(KeepRatio+NewRatio)
	MutationFactor=0.4
	Img=Img.cpu()
	OrigData= np.tile(Img, (GenomeSize,1,1,1) )

	Positions=np.zeros((GenomeSize,StickerNum*2))
	Sizes=np.zeros((GenomeSize,StickerNum*2))
	NewPositions=np.zeros((GenomeSize,StickerNum*2))
	NewSizes=np.zeros((GenomeSize,StickerNum*2))
	
	#generate Initial Genome
	for i in range(GenomeSize):
		NewPositions[i,:]=np.random.uniform(0,Size[1],StickerNum*2)
		NewSizes[i,:]=np.random.uniform(MinStickersize,MaxStickerSize,StickerNum*2)
	
	
	for St in range(NumSteps):
            Positions=NewPositions
            Sizes=NewSizes
            
            
            #put the sticker on the image:
            StickerDataNumpy=np.copy(OrigData)

            for i in range(GenomeSize):
                  for s in range(StickerNum):
                      StickerDataNumpy[i, :, int(Positions[i,2*s]):int(Positions[i,2*s]+Sizes[i,2*s]),int(Positions[i,(2*s)+1]):int(Positions[i,(2*s)+1]+Sizes[i,(2*s)+1])]=StickerColor[s]
            StickerData=torch.tensor(StickerDataNumpy).cuda()
            for i in range(GenomeSize):
                  StickerData[i,:,:,:]=Norm(StickerData[i,:,:,:])
            Pred  =  Net(StickerData)
            Pred=F.softmax(Pred,-1).cpu().detach().numpy()
            
            Weights=Pred[:,GTLabel]-Pred[:,GoalClass]
            if np.amin(Weights)<MinWeight:
                Minindex=np.argmin(Weights)
                MinWeight=np.amin(Weights)
                MinPos=Positions[Minindex,:]
                MinSize=Sizes[Minindex,:]
                MindPred=Pred[Minindex,:]
                AttackedImg=StickerDataNumpy[Minindex,:,:,:]
            #order the Population
            Indices=range(GenomeSize)
            Weights, Indices = zip(*sorted(zip(Weights, Indices)))
            KeptIndices=Indices[0:int(KeepRatio*GenomeSize)]
            GeneratedIndices=int((1.0-NewRatio)*GenomeSize)
            NewPositions=np.zeros((GenomeSize,2*StickerNum))
            NewSizes=np.zeros((GenomeSize,2*StickerNum))
            #elitism - keep the best elements
            for a in range(len(KeptIndices)):
                NewPositions[a,:]=Positions[KeptIndices[a],:]
                NewSizes[a,:]=Sizes[KeptIndices[a],:]
            #crossover for the generated ones
            for a in range(len(KeptIndices),GeneratedIndices):
                #select two samples
                Indices=np.random.choice(range(len(KeptIndices)), 2, replace=False)
                #select point of the crossover
                CrossPoint=np.random.randint(0,(2*StickerNum)+1)
                NewPositions[a,0:CrossPoint]=Positions[KeptIndices[Indices[0]]][0:CrossPoint]
                NewPositions[a,CrossPoint:2*StickerNum]=Positions[KeptIndices[Indices[1]]][CrossPoint:2*StickerNum]
                NewSizes[a,0:CrossPoint]=Sizes[KeptIndices[Indices[0]]][0:CrossPoint]
                NewSizes[a,CrossPoint:2*StickerNum]=Sizes[KeptIndices[Indices[1]]][CrossPoint:2*StickerNum]
            #rest is new
            for a in range(GeneratedIndices,GenomeSize):
                NewPositions[a,:]=np.random.uniform(0,Size[1],2*StickerNum)
                NewSizes[a,:]=np.random.uniform(MinStickersize,MaxStickerSize,2*StickerNum)

            #random mutation 
            for a in range(len(KeptIndices),GenomeSize):
                if np.random.uniform()<MutationFactor:
                      NewPositions[a,:]+=np.random.normal(0,3,2*StickerNum)
                      NewSizes[a,:]+=np.random.normal(0,3,2*StickerNum)
                      for i in range(2*StickerNum):
                            if NewSizes[a,i]>MaxStickerSize:
	                            NewSizes[a,i]=MaxStickerSize
                            if NewSizes[a,i]<MinStickersize:
	                            NewSizes[a,i]=MinStickersize
            
	if  MinWeight<-0.1:
	#if the attack is succesfull write it:
            f = open("attacks/"+str(ImageIndex).zfill(5)+"_params.txt", "w")
            f.write("GT Index: "+str(GTLabel)+"\n")
            f.write("Attack Index: "+str(GoalClass)+"\n")
            f.write("Original dist: "+str(OrigPred)+"\n")
            f.write("Attacked dist: "+str(MindPred)+"\n")
            f.write("Attack positions: "+str(MinPos)+"\n")
            f.write("AttackSizes: "+str(MinSize)+"\n")
            f.close()
            #save images
            
            cv2.imwrite("attacks/"+str(ImageIndex).zfill(5)+'_input_img.png', np.transpose(Img.numpy()[0,:,:,:]*255, (1,2, 0)) )
            cv2.imwrite("attacks/"+str(ImageIndex).zfill(5)+'attacked_img.png', np.transpose(AttackedImg*255, (1,2, 0)) )
	print("Initial Value: "+str(InitialValue)+" Optimized Value: "+str(MinWeight))



train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('/home/horan/Data/cifar10_data', download=True, train=True, transform=transforms.Compose([
transforms.ToTensor()])), 
batch_size = batch_size, shuffle=False)
#mean = [0.4914, 0.4822, 0.4465]
#std = [0.2023, 0.1994, 0.2010]

Norm=transforms.Normalize(([0.4914, 0.4822, 0.4465]), ([0.2023, 0.1994, 0.2010]))

#pretrained models are taken from here:
#https://github.com/huyvnphan/PyTorch_CIFAR10
#load pretrained model
Net = mobilenetv2.mobilenet_v2(pretrained=True)
Net.cuda()
Net.eval()
#go through batch
for batch_id, (data, label) in enumerate(train_loader):
       data=data.cuda()
       NormalizedImg=Norm(data[0,:,:,:]) 
       label=label.cuda()       
       response = Net( NormalizedImg.unsqueeze(0) )
       response=F.softmax(response,-1)[0]

       predlabel = response.argmax().cpu().detach().numpy()
       gtlabel=label.cpu().numpy()[0]
       #check if sample if correctly calssified
       if predlabel==gtlabel:
          #attack sample
          GoalClass=np.random.randint(0,NumClasses-2)
          if gtlabel<=GoalClass:
                GoalClass+=1
                GeneticStickerAttack(data, Net, response, gtlabel, GoalClass,batch_id)

