import torchvision, torchreid, torch

class Feature_Extractor_Torchreid():    
    def __init__(self):
        self.extractor = torchreid.utils.FeatureExtractor(
            model_name='osnet_ain_x1_0',
            model_path='feature_extractor_weights/osnet_ain_ms_m_c.pth.tar',
            device='cuda'
        )
        self.totensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        
    def prdict_multi_input(self, objs, ids): # images(numpy), ids(numpy)
        if len(objs) > 0:
            objs = torch.FloatTensor(objs/255).permute(0, 3, 1, 2)
            features = self.extractor(objs).detach().to('cpu')
        else : 
            features = []
        return features, ids
        