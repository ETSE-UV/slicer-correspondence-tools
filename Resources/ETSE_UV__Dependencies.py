import importlib
import slicer
from slicer.ScriptedLoadableModule import *

def ensure_packages(required_packages, interactive=False, module_name="This module"):
    """
    required_packages: list of (import_name, pip_name)
    Example: [("trimesh", "trimesh"), ("sklearn", "scikit-learn")]
    """
    missing = []

    for import_name, pip_name in required_packages:
        try:
            importlib.import_module(import_name)
        except ModuleNotFoundError:
            missing.append(pip_name)

    if not missing:
        return

    if interactive:
        ok = slicer.util.confirmYesNoDisplay(
            f"{module_name} needs to install the following Python packages into "
            f"Slicer's Python environment:\n\n  - "
            + "\n  - ".join(missing)
            + "\n\nContinue?",
            windowTitle="Install Python dependencies",
        )
        if not ok:
            raise RuntimeError("User cancelled dependency installation.")

    progress_dialog = slicer.util.createProgressDialog(
        windowTitle="Installing...",
        labelText="Installing Python packages. This may take a minute...",
        maximum=0,
    )
    slicer.app.processEvents()

    try:
        slicer.util.pip_install(" ".join(missing))
    finally:
        progress_dialog.close()

    failed = []
    for import_name, pip_name in required_packages:
        try:
            importlib.import_module(import_name)
        except Exception as e:
            failed.append(f"{pip_name}: {e}")

    if failed:
        raise RuntimeError(
            "Some Python packages could not be imported after installation. "
            "You may need to restart Slicer.\n\n"
            + "\n".join(failed)
        )