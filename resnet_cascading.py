import torch
from torchvision import models
import re
import os


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = models.resnet18(pretrained=True)
model.to(device)
model = model.eval()

print(model)

# model.load_state_dict(torch.load('./models/Mixed_7c.pth'))


layer_randomization_order = ['Logits', 'Mixed_7c',
                             'Mixed_7b', 'Mixed_7a',
                             'Mixed_6e', 'Mixed_6d',
                             'Mixed_6c', 'Mixed_6b',
                             'Mixed_6a', 'Mixed_5d',
                             'Mixed_5c', 'Mixed_5b',
                             'Conv2d_4a_3x3', 'Conv2d_3b_1x1',
                             'Conv2d_2b_3x3', 'Conv2d_2a_3x3',
                             'Conv2d_1a_3x3']


pretrained_stat = './models/resnet/pretrained.pth'
if not os.path.exists(pretrained_stat):
    torch.save(model.state_dict(), './models/resnet/pretrained.pth')
# print(model.fc.weight)
model.load_state_dict(torch.load('./models/resnet/pretrained.pth'))


# begin randomization
for i, layer_name in enumerate(layer_randomization_order):
    weight_path = './models/' + layer_name + '.pth'
    if i == 0:
        torch.nn.init.normal_(model.fc.weight)
        # torch.save(model.state_dict(), weight_path)
    else:
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path))
        print("Cascading reinitialization up to on layer {}".format(layer_name))
        layer_weight_order = [model.fc.weight, model.Mixed_7c.state_dict(),
                              model.Mixed_7b.state_dict(), model.Mixed_7a.state_dict(),
                              model.Mixed_6e.state_dict(), model.Mixed_6d.state_dict(),
                              model.Mixed_6c.state_dict(), model.Mixed_6b.state_dict(),
                              model.Mixed_6a.state_dict(), model.Mixed_5d.state_dict(),
                              model.Mixed_5b.state_dict(), model.Mixed_5b.state_dict(),
                              model.Conv2d_4a_3x3.state_dict(), model.Conv2d_3b_1x1.state_dict(),
                              model.Conv2d_2a_3x3.state_dict(), model.Conv2d_2a_3x3.state_dict(),
                              model.Conv2d_1a_3x3.state_dict()]

        for k, v in layer_weight_order[i].items():
            if re.match(r'.*weight$', k):
                torch.nn.init.normal_(v)

        #print(layer_weight_order[i])
        #torch.nn.init.normal_(layer_weight_order[i])
        # torch.save(model.state_dict(), weight_path)
