import torch
from torch import nn
from typing import List

class EBM(nn.Module):
    def __init__(self):
        super().__init__() 
    
    def calculate_energy(self, prefix: str, generations: List[str]):
        return NotImplementedError()

class CompositeEBM(EBM):
    def __init__(self, energy_models: List[EBM], weights: List[float]):
        super().__init__()

        self.energy_models = nn.ModuleList(energy_models)
        self.weights = weights

    @property
    def device(self):
        """Get device from model parameters"""
        return next(self.energy_models.parameters()).device
    
    def calculate_energy(self, prefix: str, generations: List[str]):
        
        all_energies = []
        for energy_model in self.energy_models:
            all_energies.append(energy_model.calculate_energy(prefix, generations))
        
        stacked_energies = torch.stack(all_energies, dim=1).float()

        weighted_sum = (stacked_energies * torch.tensor(self.weights, device=stacked_energies.device)).sum(dim=1)
        
        return weighted_sum, stacked_energies
        