def clean_parameter_name(name: str) -> str:
    """Standardizes parameter names across the entire project."""
    # This ensures "model.layers.0.weight" becomes "layers.0.weight"
    # and handles potential DistributedDataParallel prefixes
    name = name.replace("module.", "").replace("student_model.", "")
    if name.startswith("model."):
        name = name[6:]
    return name