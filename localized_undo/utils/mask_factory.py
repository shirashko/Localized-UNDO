import torch
from typing import Dict, List, Optional, Iterator, Tuple
import numpy as np
import re

MASK_TYPES = ["global", "random", "delta_mask"]


class MaskFactory:
    """
    Factory to generate masks for localized weight corruption.
    Includes explicit debugging to verify parameter matching and exclusions.
    """

    @staticmethod
    def _get_clean_name(name: str) -> str:
        """Centralized naming cleanup logic."""
        return name.replace("module.", "").replace("student_model.", "")

    @staticmethod
    def _target_parameters(model: torch.nn.Module) -> Iterator[Tuple[str, torch.Tensor]]:
        """Yields cleaned names and parameters that require gradients."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                yield MaskFactory._get_clean_name(name), param

    @staticmethod
    def _is_excluded(clean_name: str, exclude_components: Optional[List[str]]) -> bool:
        """Checks if a component name matches any exclusion strings."""
        if not exclude_components:
            return False
        return any(comp in clean_name for comp in exclude_components)

    @staticmethod
    def debug_naming_mismatch(model: torch.nn.Module, ref_model: torch.nn.Module):
        """Verifies if parameter names match between model and reference."""
        model_names = set(MaskFactory._get_clean_name(n) for n, _ in model.named_parameters())
        ref_names = set(MaskFactory._get_clean_name(n) for n, _ in ref_model.named_parameters())

        only_in_model = model_names - ref_names
        only_in_ref = ref_names - model_names

        print("\n" + "=" * 50)
        print("NAMING DEBUG AUDIT")
        print("=" * 50)
        if not only_in_model and not only_in_ref:
            print("[SUCCESS] All parameter names match perfectly.")
        else:
            if only_in_model: print(f"[WARNING] Only in Model: {list(only_in_model)[:5]}...")
            if only_in_ref: print(f"[WARNING] Only in Reference: {list(only_in_ref)[:5]}...")
        print("=" * 50 + "\n")

    @staticmethod
    def get_mask(
            mask_type: str,
            model: torch.nn.Module,
            percentile: float = 0.1,
            ref_model: Optional[torch.nn.Module] = None,
            exclude_components: Optional[List[str]] = None,
            device: Optional[torch.device] = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Main entry point for mask generation."""
        if mask_type is None or mask_type == "global":
            return None

        if mask_type not in MASK_TYPES:
            raise ValueError(f"Invalid mask_type: {mask_type}. Must be one of {MASK_TYPES}.")

        if mask_type == "delta_mask" and ref_model is None:
            raise ValueError("delta_mask requires a ref_model.")

        if mask_type == "random":
            return MaskFactory._generate_random_mask(model, percentile, exclude_components)

        if mask_type == "delta_mask":
            return MaskFactory._generate_delta_mask(model, ref_model, percentile, exclude_components, device)

        return None

    @staticmethod
    @torch.no_grad()
    def _generate_delta_mask(
            model: torch.nn.Module,
            ref_model: torch.nn.Module,
            percentile: float,
            exclude_components: Optional[List[str]],
            device: Optional[torch.device]
    ) -> Dict[str, torch.Tensor]:
        """Generates delta mask with detailed skip/include logging."""
        masks = {}
        ref_params = {MaskFactory._get_clean_name(n): p for n, p in ref_model.named_parameters()}

        stats = {"included": 0, "excluded_rule": 0, "not_in_ref": 0, "not_2d": 0}
        debug_excluded = []

        for clean_name, param in MaskFactory._target_parameters(model):
            masks[clean_name] = torch.zeros_like(param.data)

            # 1. Check if parameter exists in reference
            if clean_name not in ref_params:
                stats["not_in_ref"] += 1
                continue

            # 2. Check exclusion rules (e.g., embed_tokens, self_attn)
            if MaskFactory._is_excluded(clean_name, exclude_components):
                stats["excluded_rule"] += 1
                debug_excluded.append(clean_name)
                continue

            # 3. Check dimensionality (only weight matrices)
            if len(param.data.shape) < 2:
                stats["not_2d"] += 1
                continue

            stats["included"] += 1
            calc_device = device if device else param.device
            diff = torch.abs(param.data - ref_params[clean_name].data.to(calc_device))

            n_elements = diff.numel()
            k = int(n_elements * (1 - percentile))

            if k < n_elements:
                threshold = torch.kthvalue(diff.flatten(), k).values
                masks[clean_name] = (diff >= threshold).float()
            else:
                masks[clean_name] = torch.ones_like(diff)

        print(f"\n[DEBUG MASK GENERATION] Params processed:")
        print(f" -> Included: {stats['included']}")
        print(f" -> Excluded by rule: {stats['excluded_rule']} (e.g., {debug_excluded[:3]})")
        print(f" -> Skipped (Not in Reference): {stats['not_in_ref']}")
        print(f" -> Skipped (Not 2D): {stats['not_2d']}")

        return masks

    @staticmethod
    @torch.no_grad()
    def _generate_random_mask(model, percentile, exclude_components) -> Dict[str, torch.Tensor]:
        masks = {}
        for clean_name, param in MaskFactory._target_parameters(model):
            if not MaskFactory._is_excluded(clean_name, exclude_components) and len(param.data.shape) >= 2:
                masks[clean_name] = (torch.rand_like(param.data) < percentile).float()
            else:
                masks[clean_name] = torch.zeros_like(param.data)
        return masks

    @staticmethod
    def analyze_mask(mask: Dict[str, torch.Tensor], model: torch.nn.Module) -> Dict[str, float]:
        """Heuristic analysis with regex check for unassigned weights."""
        total_params = 0
        masked_params = 0
        layer_stats = {}
        unmatched_active = []

        for clean_name, m in mask.items():
            n_total = m.numel()
            n_masked = (m > 0).sum().item()
            total_params += n_total
            masked_params += n_masked

            layer_match = re.search(r'layers\.(\d+)\.', clean_name)
            if layer_match:
                l_idx = int(layer_match.group(1))
                if l_idx not in layer_stats:
                    layer_stats[l_idx] = {"total": 0, "masked": 0}
                layer_stats[l_idx]["total"] += n_total
                layer_stats[l_idx]["masked"] += n_masked
            elif n_masked > 0:
                unmatched_active.append(clean_name)

        if unmatched_active:
            print(f"\n[WARNING] Found masked weights outside 'layers.X' structure: {unmatched_active}")

        overall_sparsity = masked_params / total_params if total_params > 0 else 0
        layer_densities = [s["masked"] / s["total"] for s in layer_stats.values() if s["total"] > 0]
        localization_entropy = -sum(d * np.log(d + 1e-9) for d in layer_densities) if layer_densities else 0

        return {
            "mask/total_sparsity": overall_sparsity,
            "mask/masked_count": masked_params,
            "mask/localization_entropy": localization_entropy,
            "mask/active_layers": len(layer_stats)
        }