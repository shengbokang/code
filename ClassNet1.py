import torch.nn as nn
import torch


class ClassNet(nn.Module) :
    def __init__(self,num_classes = 26):
        super(ClassNet,self).__init__()
        self.f1 = features()
        self.f2 = features()
        self.f3 = features()
        self.f4 = features()

        self.att = nn.Conv2d(256 * 4,256,kernel_size = 1)

        self.c = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.Conv2d(256,256,kernel_size = 3,padding = 1),
            nn.ReLU(inplace  = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
        )

        self.classifier = nn.Sequential( 
            #6 x 6 x 256
            nn.Dropout(),
            nn.Linear(256 * 6 * 6,4096),
            #4096
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            #4096
            nn.ReLU(inplace = True),
            nn.Linear(4096,num_classes),
        )

    def forward(self,x):

        x0 = self.f1(x[:,:,0:226,:])    #N x C x H x W
        x1 = self.f2(x[:,:,112:338,:])    #N x C x H x W
        x2 = self.f3(x[:,:,227:453,:])    #N x C x H x W
        x3 = self.f4(x[:,:,339:565,:])    #N x C x H x W
        f = torch.cat((x0,x1,x2,x3),dim = 1)

        f = self.att(f)
        #13 x 13 x 256
        
        f = self.c(f)
        
        f = f.view(f.size(0),256 * 6 * 6)
        f = self.classifier(f)
        return f



class features(nn.Module):
    def __init__(self) :
        super(features,self).__init__()
        self.features = nn.Sequential(
            #227 x 227 x 3
            nn.Conv2d(3,64,kernel_size = 11, stride = 4, padding = 2),
            nn.ReLU(inplace = True),
            # 56 x 56 x 64
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            # 27 x 27 x 64
            nn.Conv2d(64,192,kernel_size = 5,padding = 2),
            nn.ReLU(inplace = True),
            #27 x 27 x 192
            nn.MaxPool2d(kernel_size = 3,stride = 2),
            #13 x 13 x 192
            nn.Conv2d(192,384,kernel_size = 3,padding = 1),
            nn.ReLU(inplace = True),
            #13 x 13 x 384
            nn.Conv2d(384,256,kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            #13 x 13 x 256
            nn.Conv2d(256,256,kernel_size = 3,padding = 1),
            nn.ReLU(inplace = True),
            #13 x 13 x 256
            #TODO:maxpool?

        )
    
    def forward(self, x):
        return self.features(x)