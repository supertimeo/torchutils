import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms
from torchsummary import summary

class FourierReduct:
    def __init__(self, device='cpu', coef=0.8):
        self.device = device
        self.coef = coef  # Coef < 1. Plus coef est petit, plus on coupe de hautes fréquences.

    def __call__(self, img: torch.Tensor): # Attend un tenseur PyTorch
        # Assurer que l'image est un tenseur et sur le bon device
        if not isinstance(img, torch.Tensor):
            # Ceci est une conversion basique, pour des images PIL/Numpy,
            # transforms.ToTensor() est généralement mieux pour la normalisation et la permutation des axes.
            # Ici, on suppose que 'img' est déjà dans un format HWC ou CHW numérique.
            img = torch.tensor(img, dtype=torch.float32)

        # S'assurer que l'image a une dimension de canal si elle est en niveaux de gris et 2D
        # Par exemple, si img est (H, W), la transformer en (1, H, W)
        # Si img est (B, H, W), la transformer en (B, 1, H, W)
        # Pour cet exemple, on suppose que l'image est au moins 3D (C, H, W) ou 4D (B, C, H, W)
        if img.ndim == 2: # H, W
            img = img.unsqueeze(0).unsqueeze(0) # Devient (1, 1, H, W)
        elif img.ndim == 3: # C, H, W
            img = img.unsqueeze(0) # Devient (1, C, H, W)
        # else: on suppose que c'est (B, C, H, W)

        img = img.to(self.device)

        # Appliquer la FFT2 sur les deux dernières dimensions
        # L'entrée 'img' est réelle. La sortie 'img_fft' sera complexe.
        img_fft = torch.fft.fft2(img, norm='ortho') # norm='ortho' est une bonne pratique

        # Décaler la fréquence zéro au centre pour le masquage
        img_fft_shifted = torch.fft.fftshift(img_fft, dim=(-2, -1))

        # Créer un masque pour réduire la fréquence (filtre passe-bas)
        _B, _C, H, W = img_fft_shifted.shape # Utiliser les dimensions du tenseur fft
        mask = torch.zeros((H, W), dtype=torch.bool, device=self.device) # Commencer par des zéros

        center_h, center_w = H // 2, W // 2
        # Le rayon est un pourcentage de la plus petite dimension de l'image
        # Un coef plus petit garde moins de fréquences (plus de "réduction")
        radius_h = int(center_h * self.coef)
        radius_w = int(center_w * self.coef) # On pourrait aussi utiliser min(H,W) * coef / 2

        # Créer un masque circulaire (ou elliptique si H != W et on utilise radius_h, radius_w)
        y, x = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
        # Masque circulaire: garde les fréquences basses
        # dist_from_center = torch.sqrt(((y - center_h)**2 + (x - center_w)**2))
        # mask[dist_from_center <= radius] = True
        # Pour un masque elliptique plus simple à indexer :
        mask[center_h - radius_h : center_h + radius_h, center_w - radius_w : center_w + radius_w] = True


        # Étendre le masque aux dimensions du batch et des canaux
        # Le masque est (H,W), on veut (B, C, H, W) pour la multiplication
        mask = mask.unsqueeze(0).unsqueeze(0).expand_as(img_fft_shifted)

        # Appliquer le masque
        img_fft_shifted_masked = img_fft_shifted * mask

        # Revenir à l'arrangement original des fréquences
        img_fft_masked = torch.fft.ifftshift(img_fft_shifted_masked, dim=(-2, -1))

        # Revenir dans l'espace spatial
        img_reconstructed = torch.fft.ifft2(img_fft_masked, norm='ortho')

        # Prendre la partie réelle de l'image reconstruite
        # La sortie de ifft2 est complexe, mais pour une entrée réelle et un filtre symétrique (approximativement ici),
        # la partie imaginaire devrait être proche de zéro.
        img_reconstructed_real = img_reconstructed.real

        img_reconstructed_real = img_reconstructed_real.to('cpu') # Renvoyer sur CPU

        # S'assurer que les dimensions de sortie correspondent à celles d'entrée si possible
        if img_reconstructed_real.shape[0] == 1 and img.ndim < 4 : # Si on a ajouté un batch dim
            img_reconstructed_real = img_reconstructed_real.squeeze(0)
        if img_reconstructed_real.shape[0] == 1 and img.ndim < 3 and img_reconstructed_real.ndim > 2 : # Si on a ajouté un channel et batch dim pour une image 2D
            img_reconstructed_real = img_reconstructed_real.squeeze(0)


        return img_reconstructed_real


class InverseColor:
    def __init__(self, device='cpu'):
        self.device = device

    def __call__(self, img: torch.Tensor):
        if not isinstance(img, torch.Tensor):
            # De même, cette conversion est basique.
            # Si l'image vient d'un fichier, elle est souvent uint8 [0, 255]
            # transforms.ToTensor() la convertit en float [0, 1]
            img = torch.tensor(img)

        img = img.to(self.device)

        # Déterminer la plage de l'image pour une inversion correcte
        # Si l'image est de type flottant et que sa valeur max est <= 1.0 (typique pour [0,1])
        if img.dtype == torch.float32 or img.dtype == torch.float64:
            if img.max() <= 1.0 and img.min() >=0.0 : # Supposition: normalisée [0,1]
                img_inverted = 1.0 - img
            else: # Supposition: flottant mais dans la plage [0, 255]
                img_inverted = 255.0 - img
        elif img.dtype == torch.uint8: # Typiquement [0, 255]
            # Attention: soustraire un uint8 de 255 peut causer des problèmes si 255 est un int Python
            # et img est un tenseur. Il est plus sûr de convertir ou d'utiliser un tenseur.
            img_inverted = 255 - img # PyTorch gère bien cela pour les types numériques
        else:
            # Comportement par défaut ou lever une erreur si le type/plage n'est pas géré
            print(f"Avertissement: type d'image {img.dtype} non géré explicitement pour l'inversion, en supposant la plage [0,255].")
            img_inverted = 255 - img

        img_inverted = img_inverted.to('cpu')
        return img_inverted.squeeze(1)