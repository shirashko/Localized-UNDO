import torch
from typing import Dict, List, Optional, Iterator, Tuple, Union
import numpy as np
import re

MASK_TYPES = ["global", "random", "delta_mask"]


class MaskFactory:
    """
    Factory to generate masks for localized weight corruption.
    Supports both 'global' (top-k across all targeted layers)
    and 'layer-wise' (equal budget per layer) distribution modes.
    """

    @staticmethod
    def _get_clean_name(name: str) -> str:
        return name.replace("module.", "").replace("student_model.", "")

    @staticmethod
    def _target_parameters(model: torch.nn.Module) -> Iterator[Tuple[str, torch.Tensor]]:
        for name, param in model.named_parameters():
            if param.requires_grad:
                yield MaskFactory._get_clean_name(name), param

    @staticmethod
    def _is_excluded(clean_name: str, exclude_components: Optional[List[str]]) -> bool:
        if not exclude_components:
            return False
        return any(comp in clean_name for comp in exclude_components)

    @staticmethod
    @torch.no_grad()
    def get_mask(
            mask_type: str,
            model: torch.nn.Module,
            percentile: float = 0.1,
            ref_model: Optional[torch.nn.Module] = None,
            exclude_components: Optional[List[str]] = None,
            device: Optional[torch.device] = None,
            distribution_mode: str = "layer-wise"  # "global" or "layer-wise"
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Main entry point for mask generation with distribution control."""
        if mask_type is None: return None

        # Logic for delta_mask with distribution control
        if mask_type == "delta_mask":
            if ref_model is None:
                raise ValueError("delta_mask requires a ref_model.")
            return MaskFactory._generate_delta_mask(
                model, ref_model, percentile, exclude_components, device, distribution_mode
            )

        # Basic implementation for random mask
        if mask_type == "random":
            return MaskFactory._generate_random_mask(model, percentile, exclude_components)

        return None

    @staticmethod
    @torch.no_grad()
    def _generate_delta_mask(
            model: torch.nn.Module,
            ref_model: torch.nn.Module,
            percentile: float,
            exclude_components: Optional[List[str]],
            device: Optional[torch.device],
            distribution_mode: str
    ) -> Dict[str, torch.Tensor]:
        masks = {}
        ref_params = {MaskFactory._get_clean_name(n): p for n, p in ref_model.named_parameters()}
        included_data = []  # Store tuples of (name, param, diff)
        total_targeted_elements = 0
        calc_device = device if device else next(model.parameters()).device

        # --- FIRST PASS: Single iteration to gather all necessary data ---
        for clean_name, param in MaskFactory._target_parameters(model):
            masks[clean_name] = torch.zeros_like(param.data)

            if clean_name not in ref_params: continue
            if MaskFactory._is_excluded(clean_name, exclude_components): continue
            if len(param.data.shape) < 2: continue

            # Calculate absolute difference and move to central device for global calculation
            diff = torch.abs(param.data - ref_params[clean_name].data.to(param.device))

            # Add tiny noise to break ties for precise Top-K selection
            diff += torch.randn_like(diff) * 1e-9

            included_data.append((clean_name, param, diff))
            total_targeted_elements += param.numel()

        total_noise_budget = int(total_targeted_elements * percentile)

        # --- SECOND PASS: Thresholding logic ---
        if distribution_mode == "global":
            # Gather all diffs to a single device to avoid cross-GPU issues
            all_diffs = torch.cat([d.flatten().to(calc_device) for _, _, d in included_data])

            k = all_diffs.numel() - total_noise_budget
            if k < all_diffs.numel():
                global_threshold = torch.kthvalue(all_diffs, max(1, k)).values
                for name, param, diff in included_data:
                    masks[name] = (diff.to(calc_device) >= global_threshold).float().to(param.device)
        else:
            num_components = len(included_data)
            budget_per_component = total_noise_budget // num_components
            for name, param, diff in included_data:
                n_elements = diff.numel()
                k_comp = n_elements - min(budget_per_component, n_elements)
                if k_comp < n_elements:
                    threshold = torch.kthvalue(diff.flatten(), max(1, k_comp)).values
                    masks[name] = (diff >= threshold).float()

        return masks

    @staticmethod
    @torch.no_grad()
    def _generate_random_mask(
            model: torch.nn.Module,
            percentile: float,
            exclude_components: Optional[List[str]],
            distribution_mode: str = "layer-wise"
    ) -> Dict[str, torch.Tensor]:
        """
        Generates a random binary mask with a fixed budget of active parameters.
        Maintains the same total parameter count as delta_mask for fair comparison.
        """
        masks = {}
        included_params = []
        total_targeted_elements = 0

        # --- FIRST PASS: Identify targeted parameters and total budget ---
        for clean_name, param in MaskFactory._target_parameters(model):
            masks[clean_name] = torch.zeros_like(param.data)
            if MaskFactory._is_excluded(clean_name, exclude_components): continue
            if len(param.data.shape) < 2: continue

            included_params.append(clean_name)
            total_targeted_elements += param.numel()

        total_noise_budget = int(total_targeted_elements * percentile)

        # --- SECOND PASS: Apply random noise based on distribution mode ---
        if distribution_mode == "global":
            # 1. Create a global flat index pool for all targeted weights
            indices = torch.arange(total_targeted_elements, device=next(model.parameters()).device)
            # 2. Randomly select specific indices based on the total budget
            perm = torch.randperm(total_targeted_elements, device=indices.device)
            selected_indices = perm[:total_noise_budget]

            # 3. Map these global indices back to the individual layer masks
            current_offset = 0
            for name in included_params:
                p_data = dict(MaskFactory._target_parameters(model))[name]
                n_el = p_data.numel()
                # Find indices that fall within the current layer's range
                layer_mask_flat = torch.zeros(n_el, device=indices.device)

                # Logical indexing to find relevant selected indices for this layer
                local_indices = selected_indices[(selected_indices >= current_offset) &
                                                 (selected_indices < current_offset + n_el)]
                layer_mask_flat[local_indices - current_offset] = 1.0
                masks[name] = layer_mask_flat.view_as(p_data)
                current_offset += n_el

        else:  # "layer-wise"
            num_components = len(included_params)
            budget_per_component = total_noise_budget // num_components

            for name in included_params:
                p_data = dict(MaskFactory._target_parameters(model))[name]
                n_el = p_data.numel()
                # Randomly shuffle indices within the specific component
                perm = torch.randperm(n_el, device=p_data.device)
                layer_mask_flat = torch.zeros(n_el, device=p_data.device)
                layer_mask_flat[perm[:min(budget_per_component, n_el)]] = 1.0
                masks[name] = layer_mask_flat.view_as(p_data)

        return masks

    @staticmethod
    def analyze_mask(mask: Dict[str, torch.Tensor]) -> Dict[str, Union[float, int]]:
        """
        Performs diagnostic analysis on the generated mask.
        Verifies sparsity and spatial distribution across layers.
        """
        total_elements = 0
        active_elements = 0
        layer_active_counts = {}

        for name, m in mask.items():
            n_total = m.numel()
            n_active = torch.sum(m > 0).item()

            total_elements += n_total
            active_elements += n_active

            # Extract layer index for spatial analysis
            layer_match = re.search(r'layers\.(\d+)\.', name)
            if layer_match:
                l_idx = int(layer_match.group(1))
                if l_idx not in layer_active_counts:
                    layer_active_counts[l_idx] = 0
                layer_active_counts[l_idx] += n_active

        actual_percentile = active_elements / total_elements if total_elements > 0 else 0

        # Calculate how 'spread out' the mask is across layers
        active_per_layer = list(layer_active_counts.values())
        layer_variance = np.var(active_per_layer) if active_per_layer else 0

        print(f"\n--- Mask Analysis ---")
        print(f"Total Targeted Params: {total_elements:,}")
        print(f"Active (Noised) Params: {active_elements:,}")
        print(f"Actual Sparsity: {actual_percentile:.4%}")
        print(f"Layer Distribution Variance: {layer_variance:.2f}")

        return {
            "mask/total_elements": total_elements,
            "mask/active_elements": active_elements,
            "mask/actual_sparsity": actual_percentile,
            "mask/layer_variance": layer_variance
        }