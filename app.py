import torch
import torch.nn as nn
from torchvision import datasets
import hydra



class Application(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Application,self).__init__()
        self.epochs = kwargs["epochs"]
        self.lr = kwargs["lr"]
        self.model_type = kwargs["model"]
        self.dataset=getattr(datasets,kwargs["dataset"])

        print(kwargs)


    def _load_data(self,*args):
        if args[0]=="train":
            self.dataset(root=self.data_dir,train=True,download=True,    
        elif args[0]=="test":

    def forward(self,x):
        pass

@hydra.main(config_path="./defaults.yaml")
def main(cfg):
    app=Application(**cfg.parameters)



if __name__=="__main__":
    main()
