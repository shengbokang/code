import torch.nn as nn
import torch
import torch.nn.functional as F
import inception_v3



class ClassNet(nn.Module) :
    def __init__(self,num_classes = 26):
        super(ClassNet,self).__init__()
        self.f1 = inception_v3.Inception3(feature=True)
        self.f2 = inception_v3.Inception3(feature=True)
        self.f3 = inception_v3.Inception3(feature=True)
        self.f4 = inception_v3.Inception3(feature=True)

        self.att = inception_v3.BasicConv2d(768 * 4,768,kernel_size = 1)

        self.Mixed_7a = inception_v3.InceptionD(768)
        self.Mixed_7b = inception_v3.InceptionE(1280)
        self.Mixed_7c = inception_v3.InceptionE(2048)
        self.fc = nn.Linear(2048, num_classes)


    def forward(self,x):

        '''x0 = self.f1(x[:,:,0:226,:])   #N x C x H x W
        x1 = self.f1(x[:,:,112:338,:])    #N x C x H x W
        x2 = self.f1(x[:,:,227:453,:])    #N x C x H x W
        x3 = self.f1(x[:,:,339:565,:])    #N x C x H x W'''

        #
                #299 x 299 x 3      
        x0 = self.f1(x[:,:,0:299,:])    #[)

        x1 = self.f2(x[:,:,150:449,:])
        x2 = self.f3(x[:,:,299:598,:])
        x3 = self.f4(x[:,:,449:748,:])

        f = torch.cat((x0,x1,x2,x3),dim = 1)

        
        #17 x 17 x (768x4)
        f = self.att(f)
        #17 x 17 x 768

        f = self.Mixed_7a(f)
        f = self.Mixed_7b(f)
        f = self.Mixed_7c(f)
        #8 x 8 x 2048

        f = F.avg_pool2d(f, kernel_size=8)
        # 1 x 1 x 2048
        f = F.dropout(f, training=self.training)

        f = f.view(f.size(0),-1)

        f = self.fc(f)
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