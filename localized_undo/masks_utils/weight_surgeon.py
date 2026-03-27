import torch
from typing import Dict

class WeightSurgeon:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        # Store original weights in CPU to save GPU memory
        self.original_weights = {n: p.data.clone().cpu() for n, p in model.named_parameters()}

    def restore(self):
        """Resets the model to its original state."""
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.original_weights:
                    p.data.copy_(self.original_weights[n].to(p.device))

    @torch.no_grad()
    def apply_masks(self, masks: Dict[str, torch.Tensor]):
        """Performs the ablation: W_new = P * W_old."""
        for name, param in self.model.named_parameters():
            clean_name = name.replace("model.", "")
            if clean_name in masks:
                # Move mask to GPU only when needed
                p_matrix = masks[clean_name].to(param.device)
                param.data = torch.mm(p_matrix, param.data)
                # Cleanup mask from GPU immediately after use to save space
                del p_matrix

