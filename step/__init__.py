import importlib

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
    print("\nCurrent status:")
    print(f"  • PyTorch: {BOLD}{RED}Not installed{RESET}")
    print(f"  • DGL: {BOLD}{RED}Not installed{RESET}")
    
    print(f"\n{BOLD}{YELLOW}Action Required:{RESET}")
    print("You must install PyTorch and DGL before using this package.")
    print(f"\n{BOLD}Quick Install (for Python 3.10, Linux, CUDA 11.7):{RESET}")
    print(f"  {BLUE}{BOLD}pip install step-kit[cu117]{RESET}")
    
    print(f"\n{BOLD}For other configurations:{RESET}")
    print("Visit the following websites for manual installation:")
    print("  🌐 https://pytorch.org/")
    print("  🌐 https://www.dgl.ai/")
    print("\nEnsure you install versions compatible with your system and CUDA version (if applicable).")
    
    print(f"\n{BOLD}{RED}Remember: This package will not function until PyTorch and DGL are properly installed.{RESET}")

try:
    torch = importlib.import_module('torch')
    dgl = importlib.import_module('dgl')
except ImportError:
    show_info()
else:
    __all__ = ["utils"]
    from .integration import crossModel  # noqa
    from .scmodel import scModel  # noqa
    from .stmodel import stModel  # noqa

