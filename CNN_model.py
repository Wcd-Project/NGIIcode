import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch

class cnn_model(nn.Module):
    def __init__(self, original_model, model_name, bit):
        super(cnn_model, self).__init__()
        if model_name == 'vgg11':
            self.features = original_model.features
            cl1 = nn.Linear(25088, 4096)
            cl1.weight = original_model.classifier[0].weight
            cl1.bias = original_model.classifier[0].bias

            cl2 = nn.Linear(4096, 4096)
            cl2.weight = original_model.classifier[3].weight
            cl2.bias = original_model.classifier[3].bias

            self.classifier = nn.Sequential(
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, bit),
            )
            self.model_name = 'vgg11'
        if model_name == 'alexnet':
            self.features = original_model.features
            cl1 = nn.Linear(256 * 6 * 6, 4096)
            cl1.weight = original_model.classifier[1].weight
            cl1.bias = original_model.classifier[1].bias

            cl2 = nn.Linear(4096, 4096)
            cl2.weight = original_model.classifier[4].weight
            cl2.bias = original_model.classifier[4].bias

            self.classifier = nn.Sequential(
                nn.Dropout(),
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Linear(4096, bit),
            )
            self.model_name = 'alexnet'

        if model_name == 'resnet34':
            original_model = models.resnet34(pretrained=True)
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(512, bit),
                nn.Tanh()
            )
            # self.linear = nn.Linear(bit, 10)
            self.model_name = 'resnet34'
            # self.tanh=nn.Tanh()
            # self.sigmoid=nn.Sigmoid()
            # self.linear = nn.Linear(bit, 10)
        if model_name == 'resnet50':
            original_model = models.resnet50(pretrained=True)
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(2048, bit),
                nn.Tanh()
            )
            # self.linear = nn.Linear(bit, 10)
            self.model_name = 'resnet50'
            # self.tanh=nn.Tanh()
            # self.sigmoid=nn.Sigmoid()
            # self.linear = nn.Linear(bit, 10)
        if  model_name == 'resnet34resnet34':


            # original_model1 = models.resnet34(pretrained=True)		
            original_model1 = models.vgg16(pretrained=True)
            self.features1 = nn.Sequential(*list(original_model1.children())[:-1])
            original_model2 = models.vgg19(pretrained=True)
            # original_model2 = models.resnet34(pretrained=True)
            self.features2 = nn.Sequential(*list(original_model2.children())[:-1])
            # self.feature=cat(self.features1,self.features2)
            self.classifier = nn.Sequential(
                nn.Linear(50176, bit),
                nn.Tanh()
            )					
            self.model_name = 'resnet34resnet34'			
			
    def forward(self, x):
        # f = self.features(x)
        if self.model_name == 'vgg11':
            f = f.view(f.size(0), -1)
        if self.model_name == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)
        if self.model_name == 'resnet34':
            f = f.view(f.size(0), -1)
        if self.model_name == 'resnet50':
            f = f.view(f.size(0), -1)
        if self.model_name == 'resnet34resnet34':
            f1 = self.features1(x)		
            f1 = f1.view(f1.size(0), -1)
            f2 = self.features2(x)	
            f2 = f2.view(f2.size(0), -1)		
            f=torch.cat((f1,f2),1)		
        y = self.classifier(f)
        return y

if __name__=="__main__":
    alexnet = models.resnet34(pretrained=True)
    print(alexnet)
    # vgg11_classifier = cnn_model(vgg11, 'vgg11', 1000)
    #
    # vgg11 = vgg11.cuda()
    # vgg11_classifier = vgg11_classifier.cuda()
    #
    # # evaluation phase
    # vgg11.eval()
    # vgg11_classifier.eval()
    #
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    #
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder('data/img/', transforms.Compose([
    #         transforms.Scale(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=1,
    # )
    #
    # criterion = nn.CrossEntropyLoss().cuda()
    # for i, (input, target) in enumerate(train_loader):
    #     input_var = Variable(input.cuda())
    #     output1 = vgg11(input_var)
    #     output2 = vgg11_classifier(input_var)
    #
    #     print(output1)
    #     print(output2)
    #
