#!/usr/bin/env python3
"""
Nunchaku Chroma Installer for ComfyUI

A simple installer that adds Chroma model support to ComfyUI with Nunchaku.
Works on both Windows and Linux.

Usage:
    python install.py /path/to/ComfyUI

Example:
    Windows: python install.py C:\\ComfyUI
    Linux:   python install.py ~/ComfyUI
"""

import argparse
import os
import platform
import shutil
import sys
from pathlib import Path


def get_common_comfyui_paths():
    """Get common ComfyUI installation paths for the current platform."""
    system = platform.system()
    home = Path.home()

    if system == "Windows":
        return [
            Path("C:/ComfyUI"),
            Path("C:/Program Files/ComfyUI"),
            home / "ComfyUI",
            home / "Documents" / "ComfyUI",
            # Standalone installer location
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "ComfyUI",
        ]
    else:  # Linux/Mac
        return [
            home / "ComfyUI",
            home / "comfyui",
            Path("/opt/ComfyUI"),
            home / ".local" / "share" / "ComfyUI",
        ]


def find_comfyui_venv(comfyui_path: Path) -> Path | None:
    """Find the Python venv inside ComfyUI."""
    system = platform.system()

    # Common venv locations
    venv_paths = [
        comfyui_path / ".venv",
        comfyui_path / "venv",
        comfyui_path / "python_embeded",  # Windows standalone
    ]

    # For standalone Windows installer
    if system == "Windows":
        resources_path = comfyui_path / "resources" / "ComfyUI"
        if resources_path.exists():
            # This is the standalone installer structure
            venv_paths.insert(0, comfyui_path / ".venv")

    for venv_path in venv_paths:
        if venv_path.exists():
            return venv_path

    return None


def find_site_packages(venv_path: Path) -> Path | None:
    """Find the site-packages directory in a venv."""
    system = platform.system()

    if system == "Windows":
        site_packages = venv_path / "Lib" / "site-packages"
    else:
        # Linux/Mac - need to find the Python version
        lib_path = venv_path / "lib"
        if lib_path.exists():
            for item in lib_path.iterdir():
                if item.name.startswith("python"):
                    site_packages = item / "site-packages"
                    if site_packages.exists():
                        return site_packages
        site_packages = venv_path / "lib" / "python3" / "site-packages"

    if site_packages.exists():
        return site_packages

    return None


def find_nunchaku_in_venv(site_packages: Path) -> Path | None:
    """Find the nunchaku package in site-packages."""
    nunchaku_path = site_packages / "nunchaku"
    if nunchaku_path.exists():
        return nunchaku_path
    return None


def find_comfyui_nunchaku(comfyui_path: Path) -> Path | None:
    """Find the ComfyUI-nunchaku custom node."""
    custom_nodes = comfyui_path / "custom_nodes" / "ComfyUI-nunchaku"
    if custom_nodes.exists():
        return custom_nodes

    # Check in resources for standalone installer
    resources_custom_nodes = comfyui_path / "resources" / "ComfyUI" / "custom_nodes" / "ComfyUI-nunchaku"
    if resources_custom_nodes.exists():
        return resources_custom_nodes

    return None


def get_source_dir() -> Path:
    """Get the source directory containing the files to install."""
    return Path(__file__).parent / "nunchaku_chroma"


def install_to_nunchaku(nunchaku_path: Path, source_dir: Path, dry_run: bool = False) -> bool:
    """Install Chroma transformer to nunchaku package."""
    print(f"\n📦 Installing to nunchaku: {nunchaku_path}")

    # Copy transformer_chroma.py
    src = source_dir / "transformer_chroma.py"
    dst = nunchaku_path / "models" / "transformers" / "transformer_chroma.py"

    if not src.exists():
        print(f"  ❌ Source file not found: {src}")
        return False

    if dry_run:
        print(f"  Would copy: transformer_chroma.py")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        print(f"  ✓ Copied: transformer_chroma.py")

    # Patch __init__.py files
    init_files = [
        (nunchaku_path / "models" / "transformers" / "__init__.py",
         "from .transformer_chroma import NunchakuChromaTransformer2DModel",
         "NunchakuChromaTransformer2DModel"),
        (nunchaku_path / "models" / "__init__.py",
         None,  # No new import needed, just add to __all__
         "NunchakuChromaTransformer2DModel"),
        (nunchaku_path / "__init__.py",
         None,
         "NunchakuChromaTransformer2DModel"),
    ]

    for init_path, import_line, export_name in init_files:
        if not init_path.exists():
            print(f"  ⚠ Warning: {init_path.name} not found")
            continue

        content = init_path.read_text()
        modified = False

        # Add import if specified and not present
        if import_line and import_line not in content:
            # Add after other imports
            lines = content.split('\n')
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.startswith("from ."):
                    insert_idx = i + 1
            lines.insert(insert_idx, import_line)
            content = '\n'.join(lines)
            modified = True

        # Add to __all__ if not present
        if export_name and export_name not in content and "__all__" in content:
            # Find __all__ and add the export
            import re
            all_match = re.search(r'(__all__\s*=\s*\[)([^\]]*)', content)
            if all_match:
                existing = all_match.group(2)
                if not existing.strip().endswith(','):
                    existing = existing.rstrip() + ','
                new_all = all_match.group(1) + f'\n    "{export_name}",' + existing
                content = content[:all_match.start()] + new_all + content[all_match.end():]
                modified = True

        if modified:
            if dry_run:
                print(f"  Would modify: {init_path.name}")
            else:
                # Backup
                backup = init_path.with_suffix('.py.bak')
                if not backup.exists():
                    shutil.copy(init_path, backup)
                init_path.write_text(content)
                print(f"  ✓ Modified: {init_path.name}")

    return True


def install_to_comfyui_nunchaku(comfyui_nunchaku_path: Path, source_dir: Path, dry_run: bool = False) -> bool:
    """Install Chroma wrapper and node to ComfyUI-nunchaku."""
    print(f"\n🎨 Installing to ComfyUI-nunchaku: {comfyui_nunchaku_path}")

    comfyui_src = source_dir / "comfyui"

    # Create directories
    wrappers_dir = comfyui_nunchaku_path / "wrappers"
    nodes_models_dir = comfyui_nunchaku_path / "nodes" / "models"

    for d in [wrappers_dir, nodes_models_dir]:
        if not d.exists():
            if dry_run:
                print(f"  Would create: {d}")
            else:
                d.mkdir(parents=True, exist_ok=True)

    # Copy wrapper
    wrapper_src = comfyui_src / "wrapper.py"
    wrapper_dst = wrappers_dir / "chroma.py"
    if wrapper_src.exists():
        if dry_run:
            print(f"  Would copy: wrappers/chroma.py")
        else:
            shutil.copy(wrapper_src, wrapper_dst)
            print(f"  ✓ Copied: wrappers/chroma.py")

    # Copy loader
    loader_src = comfyui_src / "loader.py"
    loader_dst = nodes_models_dir / "chroma.py"
    if loader_src.exists():
        if dry_run:
            print(f"  Would copy: nodes/models/chroma.py")
        else:
            shutil.copy(loader_src, loader_dst)
            print(f"  ✓ Copied: nodes/models/chroma.py")

    # Copy LoRA loader
    nodes_lora_dir = comfyui_nunchaku_path / "nodes" / "lora"
    if not nodes_lora_dir.exists():
        if dry_run:
            print(f"  Would create: {nodes_lora_dir}")
        else:
            nodes_lora_dir.mkdir(parents=True, exist_ok=True)

    lora_src = comfyui_src / "lora.py"
    lora_dst = nodes_lora_dir / "chroma.py"
    if lora_src.exists():
        if dry_run:
            print(f"  Would copy: nodes/lora/chroma.py")
        else:
            shutil.copy(lora_src, lora_dst)
            print(f"  ✓ Copied: nodes/lora/chroma.py")

    # Patch __init__.py
    init_path = comfyui_nunchaku_path / "__init__.py"
    if init_path.exists():
        content = init_path.read_text()
        modified = False

        # Check if Chroma DiT loader is registered
        if "NunchakuChromaDiTLoader" not in content:
            registration = '''
try:
    from .nodes.models.chroma import NunchakuChromaDiTLoader
    NODE_CLASS_MAPPINGS["NunchakuChromaDiTLoader"] = NunchakuChromaDiTLoader
except ImportError:
    logger.exception("Node `NunchakuChromaDiTLoader` import failed:")
'''
            # Find last NODE_CLASS_MAPPINGS assignment
            if "NODE_CLASS_MAPPINGS[" in content:
                last_idx = content.rfind("NODE_CLASS_MAPPINGS[")
                end_line = content.find('\n', last_idx)
                if end_line != -1:
                    if dry_run:
                        print(f"  Would add Chroma DiT loader registration to __init__.py")
                    else:
                        content = content[:end_line+1] + registration + content[end_line+1:]
                        modified = True
                        print(f"  ✓ Added Chroma DiT loader registration to __init__.py")

        # Check if Chroma LoRA loader is registered
        if "NunchakuChromaLoraLoader" not in content:
            lora_registration = '''
try:
    from .nodes.lora.chroma import NunchakuChromaLoraLoader, NunchakuChromaLoraStack
    NODE_CLASS_MAPPINGS["NunchakuChromaLoraLoader"] = NunchakuChromaLoraLoader
    NODE_CLASS_MAPPINGS["NunchakuChromaLoraStack"] = NunchakuChromaLoraStack
except ImportError:
    logger.exception("Nodes `NunchakuChromaLoraLoader` and `NunchakuChromaLoraStack` import failed:")
'''
            # Find last NODE_CLASS_MAPPINGS assignment
            if "NODE_CLASS_MAPPINGS[" in content:
                last_idx = content.rfind("NODE_CLASS_MAPPINGS[")
                end_line = content.find('\n', last_idx)
                if end_line != -1:
                    if dry_run:
                        print(f"  Would add Chroma LoRA loader registration to __init__.py")
                    else:
                        content = content[:end_line+1] + lora_registration + content[end_line+1:]
                        modified = True
                        print(f"  ✓ Added Chroma LoRA loader registration to __init__.py")

        if modified:
            init_path.write_text(content)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Install Nunchaku Chroma support to ComfyUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Windows:  python install.py C:\\ComfyUI
    Linux:    python install.py ~/ComfyUI
    Dry run:  python install.py --dry-run /path/to/ComfyUI
"""
    )
    parser.add_argument(
        "comfyui_path",
        nargs="?",
        help="Path to ComfyUI installation"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--auto-detect",
        action="store_true",
        help="Try to auto-detect ComfyUI installation"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  Nunchaku Chroma Installer for ComfyUI")
    print("=" * 60)
    print(f"  Platform: {platform.system()}")

    # Find ComfyUI path
    comfyui_path = None

    if args.comfyui_path:
        comfyui_path = Path(args.comfyui_path).expanduser().resolve()
    elif args.auto_detect:
        print("\n🔍 Auto-detecting ComfyUI installation...")
        for path in get_common_comfyui_paths():
            if path.exists():
                print(f"  Found: {path}")
                comfyui_path = path
                break

    if comfyui_path is None:
        print("\n❌ ComfyUI path not specified.")
        print("\nUsage:")
        print("  python install.py /path/to/ComfyUI")
        print("\nOr use --auto-detect to search common locations:")
        print("  python install.py --auto-detect")
        sys.exit(1)

    if not comfyui_path.exists():
        print(f"\n❌ ComfyUI path does not exist: {comfyui_path}")
        sys.exit(1)

    print(f"\n📁 ComfyUI path: {comfyui_path}")

    # Find venv and site-packages
    venv_path = find_comfyui_venv(comfyui_path)
    if venv_path is None:
        print("\n❌ Could not find ComfyUI's Python environment")
        print("  Looked for: .venv, venv, python_embeded")
        sys.exit(1)

    print(f"  Venv: {venv_path}")

    site_packages = find_site_packages(venv_path)
    if site_packages is None:
        print("\n❌ Could not find site-packages in venv")
        sys.exit(1)

    print(f"  Site-packages: {site_packages}")

    # Find nunchaku
    nunchaku_path = find_nunchaku_in_venv(site_packages)
    if nunchaku_path is None:
        print("\n❌ Nunchaku not found in ComfyUI's environment")
        print("  Please install nunchaku first")
        sys.exit(1)

    print(f"  Nunchaku: {nunchaku_path}")

    # Find ComfyUI-nunchaku
    comfyui_nunchaku = find_comfyui_nunchaku(comfyui_path)
    if comfyui_nunchaku is None:
        print("\n⚠ ComfyUI-nunchaku custom node not found")
        print("  Will only install to nunchaku package")
    else:
        print(f"  ComfyUI-nunchaku: {comfyui_nunchaku}")

    # Get source directory
    source_dir = get_source_dir()
    if not source_dir.exists():
        print(f"\n❌ Source directory not found: {source_dir}")
        sys.exit(1)

    print(f"  Source: {source_dir}")

    if args.dry_run:
        print("\n⚠ DRY RUN - No changes will be made")

    # Install
    success = True

    success &= install_to_nunchaku(nunchaku_path, source_dir, dry_run=args.dry_run)

    if comfyui_nunchaku:
        success &= install_to_comfyui_nunchaku(comfyui_nunchaku, source_dir, dry_run=args.dry_run)

    if success:
        print("\n" + "=" * 60)
        if args.dry_run:
            print("  ✓ Dry run complete - no changes made")
        else:
            print("  ✓ Installation complete!")
            print("\n  Next steps:")
            print("  1. Restart ComfyUI")
            print("  2. Place your quantized Chroma model in:")
            print(f"     {comfyui_path / 'models' / 'diffusion_models'}")
            print("  3. Use the 'Nunchaku Chroma DiT Loader' node")
        print("=" * 60)
    else:
        print("\n❌ Installation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
