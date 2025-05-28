import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms
from torchsummary import summary

class FourierReduct:
    def __init__(self, device, coef=0.8):
        self.device = device
        self.coef = coef
    
    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)

        img = img.to(self.device)
        # Convertir l'image en complexe
        img = torch.view_as_complex(img)
        # Appliquer la FFT
        img_fft = torch.fft.fft2(img)
        # Créer un masque pour réduire la fréquence
        B, C, H, W = img_fft.shape
        mask = torch.ones((H, W), dtype=torch.bool, device=self.device)
        center_h, center_w = H // 2, W // 2
        radius = int(min(H, W) * self.coef)
        
        # Créer un masque circulaire
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        mask = ((y - center_h) ** 2 + (x - center_w) ** 2) <= radius ** 2
        mask = mask.expand(B, C, H, W)
        
        # Appliquer le masque
        img_fft = img_fft * mask
        # Revenir dans l'espace spatial
        img = torch.fft.ifft2(img_fft)
        # Convertir en réel
        img = torch.view_as_real(img).abs()
        img = img.to('cpu')
        return img



class InverseColor:
    def __init__(self, device):
        self.device = device
    
    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)
        img = img.to(self.device)
        img = 255 - img
        img = img.to('cpu')
        return img