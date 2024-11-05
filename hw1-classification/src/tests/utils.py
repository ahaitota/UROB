import importlib.util

def load_module(src_dir: str, module_name: str) -> object:
    """Dynamically load a Python module from a specified directory.

    Args:
        src_dir (str): The directory containing the module file.
        module_name (str): The name of the module to be loaded (without the .py extension).

    Returns:
        object: The loaded module object.

    Note:
        This function requires that the module file exists in the specified directory.
    """
    module_path = f"{src_dir}/{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module
