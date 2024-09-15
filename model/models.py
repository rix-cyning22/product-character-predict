from torch import nn, stack
import torchvision.models as models

class TaskModel(nn.Module):
    def __init__(self, task_input_channel, task_output_channel):
        super().__init__()
        self.feature_extractor = None 
        self.shared_params = None 
        self.task_layers = nn.ModuleList([nn.Linear(task_input_channel, task_output_channel) for _ in range(8)])
        self.evaluator = None
    
    def forward(self, x, masks):
        x_mask = []
        x = self.feature_extractor(x)
        x = self.shared_params(x)
        masks = masks.tolist()
        for i in range(x.shape[0]): 
            p = self.task_layers[masks[i]](x[i])
            x_mask.append(p) 
        x_mask = stack(x_mask).squeeze()
        return self.evaluator(x_mask)
        
class SimpleTask(TaskModel):
    def __init__(self):
        super().__init__(128, 64)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.shared_params = nn.Sequential(
            nn.Linear(246016, 128),
            nn.ReLU()
        )
        self.evaluator = nn.Sequential(
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.ReLU()
        )


class ResidualTask(TaskModel):
    def __init__(self):
        super().__init__(512, 1)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False
        layers = list(resnet.children())[:-2] 
        for layer in layers[-10:]:
            for param in layer.parameters():
                param.requires_grad = True
        self.feature_extractor = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.shared_params = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.evaluator = nn.Sequential(
            nn.ReLU()
        )


