import torch
from typing import List, Tuple, Optional


class DirectionalMaskFactory:
    """
    Implements Weight Surgery based on 'Watch the Weights' (Zhong & Raghunathan, 2025).
    Focuses specifically on MLP down_proj matrices.
    """

    @staticmethod
    @torch.no_grad()
    def compute_projection_mask(
            current_param: torch.Tensor,
            reference_param: torch.Tensor,
            k: int,
            layer_name: str = "",
            plot_path: Optional[str] = None
    ) -> Tuple[torch.Tensor, float]:
        # 1. Compute Delta W
        delta_w = current_param - reference_param.to(current_param.device)

        # 2. SVD Decomposition
        # Using .float() to ensure precision during SVD, then casting back
        u, s, vh = torch.linalg.svd(delta_w.float(), full_matrices=False)

        # 3. Calculate Explained Variance
        s_sq = s ** 2
        total_variance = torch.sum(s_sq)
        explained_variance = torch.sum(s_sq[:k]) / total_variance

        # 4. Construct Projection Matrix P = I - UU^T
        u_top = u[:, :k]
        d_model = u_top.shape[0]
        identity = torch.eye(d_model, device=current_param.device)
        projection_matrix = identity - torch.mm(u_top, u_top.t())

        return projection_matrix.to(current_param.dtype), explained_variance.item()

    @staticmethod
    def is_target_layer(name: str, exclusions: List[str]) -> bool:
        """
        Strictly targets MLP down_projection matrices.
        In Gemma/Llama, 'down_proj' is the linear layer that projects
        back to the residual stream.
        """
        is_excluded = any(exc in name for exc in exclusions)
        # Targeted only at the MLP output (down_proj)
        is_down_proj = "down_proj" in name
        return is_down_proj and not is_excluded

    @staticmethod
    def compute_random_mask(param: torch.Tensor, k: int) -> torch.Tensor:
        """Projects weights onto the null-space of k random directions."""
        d_out = param.shape[0]
        # Generate random orthonormal vectors using QR decomposition
        random_matrix = torch.randn(d_out, k, device=param.device)
        q, _ = torch.linalg.qr(random_matrix)  # q represents k random orthogonal directions

        identity = torch.eye(d_out, device=param.device)
        projection_matrix = identity - torch.mm(q, q.t())
        return projection_matrix