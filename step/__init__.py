import importlib
import pkg_resources

def is_module_installed(module_name):
    try:
        module = importlib.import_module(module_name)
        version = pkg_resources.get_distribution(module_name).version
        return True, version
    except (ImportError, pkg_resources.DistributionNotFound):
        return False, None

def show_info():
    # ANSI escape codes for colors and bold text
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    print(f"\n{BOLD}{GREEN}🎉 Thank you for installing step! 🎉{RESET}")
    print(f"\n{BOLD}{RED}Critical Requirement:{RESET}")
    print(f"This package {BOLD}requires{RESET} {BOLD}PyTorch{RESET} and {BOLD}DGL{RESET} to function.")
    print(f"{BOLD}The package will not work without these dependencies.{RESET}")

    pytorch_installed, pytorch_version = is_module_installed('torch')
    dgl_installed, dgl_version = is_module_installed('dgl')

    print("\nCurrent status:")
    if pytorch_installed:
        print(f"  • PyTorch: {BOLD}{GREEN}Installed{RESET} (version: {pytorch_version})")
    else:
        print(f"  • PyTorch: {BOLD}{RED}Not installed{RESET}")

    if dgl_installed:
        print(f"  • DGL: {BOLD}{GREEN}Installed{RESET} (version: {dgl_version})")
    else:
        print(f"  • DGL: {BOLD}{RED}Not installed{RESET}")

    if not pytorch_installed or not dgl_installed:
        print(f"\n{BOLD}{YELLOW}Action Required:{RESET}")
        print("You must install the missing dependencies before using this package.")
        print(f"\n{BOLD}Quick Install (for Python 3.10 & 3.11, CUDA 11.7):{RESET}")
        print(f"  {BLUE}{BOLD}pip install step-kit[cu117]{RESET}")
        
        print(f"\n{BOLD}For other configurations:{RESET}")
        print("Visit the following websites for manual installation:")
        print("  🌐 https://pytorch.org/")
        print("  🌐 https://www.dgl.ai/")
        print("\nEnsure you install versions compatible with your system and CUDA version (if applicable).")
        
        print(f"\n{BOLD}{RED}Remember: This package will not function until PyTorch and DGL are properly installed.{RESET}")
    else:
        print(f"\n{BOLD}{GREEN}All required dependencies are installed. You're ready to use the package!{RESET}")

# In your main code:
pytorch_installed, _ = is_module_installed('torch')
dgl_installed, _ = is_module_installed('dgl')

if not pytorch_installed or not dgl_installed:
    show_info()
else:
    __all__ = ["utils"]
    from .integration import crossModel  # noqa
    from .scmodel import scModel  # noqa
    from .stmodel import stModel  # noqa

