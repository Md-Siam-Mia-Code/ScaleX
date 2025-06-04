# patches.py

import sys
import types
import importlib.machinery  # For SourceFileLoader

# No top-level torchvision imports here; they will be handled within the function.


def apply_torchvision_patches():

    # These will hold the imported modules, local to this function's scope
    tv_transforms_module = None
    TF_functional_module = None  # This will be our torchvision.transforms.functional

    try:
        # Import the necessary modules directly here.
        # These module objects will be local to this function's execution.
        import torchvision.transforms  # Import the parent module
        from torchvision.transforms import (
            functional as TF_local,
        )  # Import the functional submodule

        tv_transforms_module = torchvision.transforms  # Assign to our local variable
        TF_functional_module = TF_local  # Assign to our local variable

    except ImportError:
        print(
            "ScaleX Patch CRITICAL ERROR: Failed to import 'torchvision.transforms' or 'torchvision.transforms.functional'."
        )
        print(
            "ScaleX Patch CRITICAL ERROR: Ensure PyTorch and TorchVision are correctly installed in the environment."
        )
        print(
            "ScaleX Patch CRITICAL ERROR: Cannot apply patches. Dependencies like 'basicsr' might fail."
        )
        return

    expected_module_fqn = "torchvision.transforms.functional_tensor"
    submodule_name_to_create = (
        "functional_tensor"  # The name of the submodule 'functional_tensor'
    )

    # This will hold our new dummy module object
    dummy_functional_tensor_module = None

    # tv_transforms_module should be the already imported 'torchvision.transforms'
    if not tv_transforms_module:
        # This case should ideally not be reached if the try-except block above succeeded.
        print(
            "ScaleX Patch ERROR: 'torchvision.transforms' module is not available after import attempt. Cannot patch."
        )
        return

    if hasattr(tv_transforms_module, submodule_name_to_create):
        dummy_functional_tensor_module = getattr(
            tv_transforms_module, submodule_name_to_create
        )
        # If it exists as an attribute, ensure it's also in sys.modules for direct import by FQN
        if expected_module_fqn not in sys.modules:
            sys.modules[expected_module_fqn] = dummy_functional_tensor_module
    else:
        dummy_functional_tensor_module = types.ModuleType(
            expected_module_fqn
        )  # Sets module.__name__
        dummy_functional_tensor_module.__file__ = (
            __file__  # Fake its origin to this patch file
        )
        dummy_functional_tensor_module.__package__ = (
            "torchvision.transforms"  # Its parent package
        )
        # Provide a dummy loader, makes it look more like a regularly imported module
        dummy_functional_tensor_module.__loader__ = (
            importlib.machinery.SourceFileLoader(expected_module_fqn, __file__)
        )

        sys.modules[expected_module_fqn] = dummy_functional_tensor_module
        setattr(
            tv_transforms_module,
            submodule_name_to_create,
            dummy_functional_tensor_module,
        )

    if not hasattr(dummy_functional_tensor_module, "rgb_to_grayscale"):
        dummy_functional_tensor_module.rgb_to_grayscale = (
            TF_functional_module.rgb_to_grayscale
        )
    else:
        # If it exists, we overwrite it to be certain it points to the correct, newer function.
        dummy_functional_tensor_module.rgb_to_grayscale = (
            TF_functional_module.rgb_to_grayscale
        )