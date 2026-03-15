import torch
from typing import Dict, List, Optional, Iterator, Tuple, Union
import numpy as np
import re
from localization_utils import clean_parameter_name

MASK_TYPES = ["global", "random", "delta_mask"]


class MaskFactory:
    """
    Factory to generate masks for localized weight corruption.
    Supports both 'global' (top-k across all targeted layers)
    and 'layer-wise' (equal budget per layer) distribution modes.
    """

    @staticmethod
    def _target_parameters(model: torch.nn.Module) -> Iterator[Tuple[str, torch.Tensor]]:
        """Yields standardized names and parameters for targeting."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                yield clean_parameter_name(name), param

    @staticmethod
    def _is_excluded(clean_name: str, exclude_components: Optional[List[str]]) -> bool:
        """Checks if a component should be excluded from masking."""
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
        if mask_type is None:
            return None

        if mask_type == "delta_mask":
            if ref_model is None:
                raise ValueError("delta_mask requires a ref_model.")
            return MaskFactory._generate_delta_mask(
                model, ref_model, percentile, exclude_components, device, distribution_mode
            )

        if mask_type == "random":
            return MaskFactory._generate_random_mask(
                model, percentile, exclude_components, distribution_mode
            )

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
        # Standardize reference parameters for lookup
        ref_params = {clean_parameter_name(n): p for n, p in ref_model.named_parameters()}

        included_data = []  # List of (name, param, diff_tensor)
        total_targeted_elements = 0
        calc_device = device if device else next(model.parameters()).device

        # --- Gather Weight Differences ---
        for clean_name, param in MaskFactory._target_parameters(model):
            masks[clean_name] = torch.zeros_like(param.data)

            if clean_name not in ref_params: continue
            if MaskFactory._is_excluded(clean_name, exclude_components): continue
            if len(param.data.shape) < 2: continue  # Focus on weights, skip biases/scalars

            # Compute absolute discrepancy
            diff = torch.abs(param.data - ref_params[clean_name].data.to(param.device))

            # Tie-breaking noise for precise Top-K
            diff += torch.randn_like(diff) * 1e-9

            included_data.append((clean_name, param, diff))
            total_targeted_elements += param.numel()

        total_noise_budget = int(total_targeted_elements * percentile)

        # --- Apply Thresholding ---
        if distribution_mode == "global":
            all_diffs = torch.cat([d.flatten().to(calc_device) for _, _, d in included_data])
            k = all_diffs.numel() - total_noise_budget

            if k < all_diffs.numel():
                global_threshold = torch.kthvalue(all_diffs, max(1, k)).values
                for name, param, diff in included_data:
                    masks[name] = (diff.to(calc_device) >= global_threshold).float().to(param.device)
        else:
            # Layer-wise: calculate budget based on the count of included components
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
        masks = {}
        included_params = []  # List of (name, param_obj)
        total_targeted_elements = 0

        for clean_name, param in MaskFactory._target_parameters(model):
            masks[clean_name] = torch.zeros_like(param.data)
            if MaskFactory._is_excluded(clean_name, exclude_components): continue
            if len(param.data.shape) < 2: continue

            included_params.append((clean_name, param))
            total_targeted_elements += param.numel()

        total_noise_budget = int(total_targeted_elements * percentile)

        if distribution_mode == "global":
            # Global random shuffling across a virtual flat tensor
            indices = torch.randperm(total_targeted_elements, device=next(model.parameters()).device)
            selected_global_indices = indices[:total_noise_budget]

            # Sort for easier mapping back to layers
            selected_global_indices, _ = torch.sort(selected_global_indices)

            current_offset = 0
            idx_ptr = 0
            for name, param in included_params:
                n_el = param.numel()
                layer_mask_flat = torch.zeros(n_el, device=param.device)

                # Find which selected global indices fall into this layer's range
                upper_bound = current_offset + n_el
                while idx_ptr < len(selected_global_indices) and selected_global_indices[idx_ptr] < upper_bound:
                    local_idx = selected_global_indices[idx_ptr] - current_offset
                    layer_mask_flat[local_idx] = 1.0
                    idx_ptr += 1

                masks[name] = layer_mask_flat.view_as(param)
                current_offset += n_el

        else:  # layer-wise
            num_components = len(included_params)
            budget_per_component = total_noise_budget // num_components

            for name, param in included_params:
                n_el = param.numel()
                perm = torch.randperm(n_el, device=param.device)
                layer_mask_flat = torch.zeros(n_el, device=param.device)
                layer_mask_flat[perm[:min(budget_per_component, n_el)]] = 1.0
                masks[name] = layer_mask_flat.view_as(param)

        return masks

    @staticmethod
    def analyze_mask(mask: Dict[str, torch.Tensor]) -> Dict[str, Union[float, int]]:
        """Diagnostic analysis of the generated mask."""
        total_elements = 0
        active_elements = 0
        layer_active_counts = {}

        for name, m in mask.items():
            n_total = m.numel()
            n_active = torch.sum(m > 0).item()
            total_elements += n_total
            active_elements += n_active

            # Spatial distribution (regex looks for 'layers.X.')
            layer_match = re.search(r'layers\.(\d+)\.', name)
            if layer_match:
                l_idx = int(layer_match.group(1))
                layer_active_counts[l_idx] = layer_active_counts.get(l_idx, 0) + n_active

        actual_sparsity = active_elements / total_elements if total_elements > 0 else 0
        active_per_layer = list(layer_active_counts.values())
        layer_variance = np.var(active_per_layer) if active_per_layer else 0

        print(f"\n--- Mask Analysis ---")
        print(f"Total Targeted Params: {total_elements:,}")
        print(f"Active (Noised) Params: {active_elements:,}")
        print(f"Actual Sparsity: {actual_sparsity:.4%}")
        print(f"Layer Distribution Variance: {layer_variance:.2f}")

        return {
            "mask/total_elements": total_elements,
            "mask/active_elements": active_elements,
            "mask/actual_sparsity": actual_sparsity,
            "mask/layer_variance": layer_variance
        }