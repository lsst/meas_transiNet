from model import rbTransiNetModel
import torch


class rbTransiNetInterface:
    ''' A class for interfacing between the LSST AP pipeline and
    an rbTransiNet model. '''

    def __init__(self, device='cpu'):
        self.model = rbTransiNetModel
        self.device = device

    def init(self, pretrained_file):

        # --- Load pre-trained model from disk
        network_data = torch.load(pretrained_file, map_location=self.device)
        self.model.load_state_dict(network_data['state_dict'], strict=True)

        # --- put model in "eval" mode and stand by
        self.model.eval()

    def prepare_input(self, x):
        ''' 
        Things like format conversion from afw.image.exposure to torch.tensor or
        stacking-up of images can happen here.
        '''
        x = x
        return x

    def infer(self, x):

        # --- Perform any required pre-processing and format conversion
        x = self.prepare_input(x)

        # --- Feed input to the network and get a score
        score = self.model(x)

        return score
