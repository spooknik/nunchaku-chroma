#!/usr/bin/env python3
"""
Nunchaku Chroma Patcher

This script patches an existing nunchaku installation to add Chroma model support.
Since Chroma reuses FLUX's CUDA kernels, no recompilation is needed - only Python
files need to be added.

Usage:
    python -m nunchaku_chroma.patcher [options]

Options:
    --verify        Only verify if Chroma support is installed
    --comfyui PATH  Also patch ComfyUI-nunchaku at the specified path
    --dry-run       Show what would be done without making changes
"""

import argparse
import importlib.util
import os
import re
import shutil
import sys
from pathlib import Path


def get_nunchaku_path():
    """Find the nunchaku installation path."""
    try:
        import nunchaku
        return Path(nunchaku.__file__).parent
    except ImportError:
        return None


def check_chroma_installed(nunchaku_path: Path) -> bool:
    """Check if Chroma support is already installed."""
    transformer_path = nunchaku_path / "models" / "transformers" / "transformer_chroma.py"
    return transformer_path.exists()


def verify_chroma_importable() -> bool:
    """Verify that NunchakuChromaTransformer2DModel is importable."""
    try:
        from nunchaku import NunchakuChromaTransformer2DModel
        return True
    except ImportError:
        return False


def get_source_files():
    """Get the source files from this package."""
    package_dir = Path(__file__).parent

    files = {
        "transformer_chroma.py": package_dir / "transformer_chroma.py",
    }

    return files


def patch_init_file(init_path: Path, additions: list[tuple[str, str]], dry_run: bool = False) -> bool:
    """
    Patch an __init__.py file to add new imports and exports.

    Parameters
    ----------
    init_path : Path
        Path to the __init__.py file
    additions : list of tuples
        List of (import_line, export_name) tuples to add
    dry_run : bool
        If True, only show what would be done

    Returns
    -------
    bool
        True if changes were made (or would be made in dry_run mode)
    """
    if not init_path.exists():
        print(f"  Warning: {init_path} does not exist")
        return False

    content = init_path.read_text()
    original_content = content
    changes_made = False

    for import_line, export_name in additions:
        # Check if import already exists
        if import_line in content or export_name in content:
            continue

        changes_made = True

        # Add import line
        # Try to find existing similar imports and add after them
        if "from ." in content:
            # Find the last "from ." import and add after it
            lines = content.split('\n')
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith("from ."):
                    insert_idx = i + 1

            # If this is a multi-line import, find the end
            while insert_idx < len(lines) and lines[insert_idx - 1].strip().endswith(','):
                insert_idx += 1

            lines.insert(insert_idx, import_line)
            content = '\n'.join(lines)
        else:
            # Just prepend
            content = import_line + '\n' + content

        # Add to __all__ if it exists
        all_match = re.search(r'__all__\s*=\s*\[([^\]]*)\]', content, re.DOTALL)
        if all_match:
            all_content = all_match.group(1)
            if export_name not in all_content:
                # Add the export
                new_all_content = all_content.rstrip()
                if new_all_content and not new_all_content.endswith(','):
                    new_all_content += ','
                new_all_content += f'\n    "{export_name}",'
                content = content[:all_match.start(1)] + new_all_content + content[all_match.end(1):]

    if changes_made:
        if dry_run:
            print(f"  Would modify: {init_path}")
        else:
            # Backup original
            backup_path = init_path.with_suffix('.py.bak')
            if not backup_path.exists():
                shutil.copy(init_path, backup_path)
            init_path.write_text(content)
            print(f"  Modified: {init_path}")

    return changes_made


def patch_nunchaku(nunchaku_path: Path, dry_run: bool = False) -> bool:
    """
    Patch nunchaku to add Chroma support.

    Parameters
    ----------
    nunchaku_path : Path
        Path to the nunchaku package directory
    dry_run : bool
        If True, only show what would be done

    Returns
    -------
    bool
        True if patching was successful
    """
    print(f"Patching nunchaku at: {nunchaku_path}")

    source_files = get_source_files()
    transformers_dir = nunchaku_path / "models" / "transformers"

    # 1. Copy transformer_chroma.py
    src = source_files["transformer_chroma.py"]
    dst = transformers_dir / "transformer_chroma.py"

    if not src.exists():
        print(f"  Error: Source file not found: {src}")
        return False

    if dry_run:
        print(f"  Would copy: {src} -> {dst}")
    else:
        shutil.copy(src, dst)
        print(f"  Copied: transformer_chroma.py")

    # 2. Patch models/transformers/__init__.py
    transformers_init = transformers_dir / "__init__.py"
    patch_init_file(
        transformers_init,
        [("from .transformer_chroma import NunchakuChromaTransformer2DModel", "NunchakuChromaTransformer2DModel")],
        dry_run=dry_run
    )

    # 3. Patch models/__init__.py
    models_init = nunchaku_path / "models" / "__init__.py"
    patch_init_file(
        models_init,
        [("", "NunchakuChromaTransformer2DModel")],  # Just add to __all__, import comes from transformers
        dry_run=dry_run
    )

    # 4. Patch nunchaku/__init__.py
    nunchaku_init = nunchaku_path / "__init__.py"
    patch_init_file(
        nunchaku_init,
        [("", "NunchakuChromaTransformer2DModel")],  # Just add to __all__
        dry_run=dry_run
    )

    return True


def patch_comfyui_nunchaku(comfyui_path: Path, dry_run: bool = False) -> bool:
    """
    Patch ComfyUI-nunchaku to add Chroma node.

    Parameters
    ----------
    comfyui_path : Path
        Path to the ComfyUI-nunchaku directory
    dry_run : bool
        If True, only show what would be done

    Returns
    -------
    bool
        True if patching was successful
    """
    print(f"Patching ComfyUI-nunchaku at: {comfyui_path}")

    package_dir = Path(__file__).parent
    comfyui_src = package_dir / "comfyui"

    # Create directories
    wrappers_dir = comfyui_path / "wrappers"
    nodes_models_dir = comfyui_path / "nodes" / "models"

    for d in [wrappers_dir, nodes_models_dir]:
        if not d.exists():
            if dry_run:
                print(f"  Would create directory: {d}")
            else:
                d.mkdir(parents=True, exist_ok=True)
                print(f"  Created directory: {d}")

    # Copy wrapper
    wrapper_src = comfyui_src / "wrapper.py"
    wrapper_dst = wrappers_dir / "chroma.py"
    if wrapper_src.exists():
        if dry_run:
            print(f"  Would copy: wrapper.py -> {wrapper_dst}")
        else:
            shutil.copy(wrapper_src, wrapper_dst)
            print(f"  Copied: wrappers/chroma.py")

    # Copy loader
    loader_src = comfyui_src / "loader.py"
    loader_dst = nodes_models_dir / "chroma.py"
    if loader_src.exists():
        if dry_run:
            print(f"  Would copy: loader.py -> {loader_dst}")
        else:
            shutil.copy(loader_src, loader_dst)
            print(f"  Copied: nodes/models/chroma.py")

    # Copy LoRA loader
    nodes_lora_dir = comfyui_path / "nodes" / "lora"
    if not nodes_lora_dir.exists():
        if dry_run:
            print(f"  Would create directory: {nodes_lora_dir}")
        else:
            nodes_lora_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Created directory: {nodes_lora_dir}")

    lora_src = comfyui_src / "lora.py"
    lora_dst = nodes_lora_dir / "chroma.py"
    if lora_src.exists():
        if dry_run:
            print(f"  Would copy: lora.py -> {lora_dst}")
        else:
            shutil.copy(lora_src, lora_dst)
            print(f"  Copied: nodes/lora/chroma.py")

    # Patch __init__.py
    init_path = comfyui_path / "__init__.py"
    if init_path.exists():
        content = init_path.read_text()

        # Check if Chroma node is already registered
        if "NunchakuChromaDiTLoader" not in content:
            # Add the node registration
            registration_code = '''
try:
    from .nodes.models.chroma import NunchakuChromaDiTLoader
    NODE_CLASS_MAPPINGS["NunchakuChromaDiTLoader"] = NunchakuChromaDiTLoader
except ImportError:
    logger.exception("Node `NunchakuChromaDiTLoader` import failed:")

try:
    from .nodes.lora.chroma import NunchakuChromaLoraLoader, NunchakuChromaLoraStack
    NODE_CLASS_MAPPINGS["NunchakuChromaLoraLoader"] = NunchakuChromaLoraLoader
    NODE_CLASS_MAPPINGS["NunchakuChromaLoraStack"] = NunchakuChromaLoraStack
except ImportError:
    logger.exception("Node `NunchakuChromaLoraLoader` import failed:")
'''
            # Find a good place to insert (after other node registrations)
            if "NODE_CLASS_MAPPINGS" in content:
                # Find the last NODE_CLASS_MAPPINGS assignment
                last_mapping = content.rfind("NODE_CLASS_MAPPINGS[")
                if last_mapping != -1:
                    # Find the end of that line
                    end_of_line = content.find('\n', last_mapping)
                    if end_of_line != -1:
                        if dry_run:
                            print(f"  Would add Chroma node registration to: {init_path}")
                        else:
                            content = content[:end_of_line+1] + registration_code + content[end_of_line+1:]
                            init_path.write_text(content)
                            print(f"  Added Chroma node registration to __init__.py")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Patch nunchaku to add Chroma model support"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify if Chroma support is installed"
    )
    parser.add_argument(
        "--comfyui",
        type=str,
        metavar="PATH",
        help="Also patch ComfyUI-nunchaku at the specified path"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    # Find nunchaku
    nunchaku_path = get_nunchaku_path()

    if nunchaku_path is None:
        print("Error: nunchaku is not installed")
        print("Please install nunchaku first: pip install nunchaku")
        sys.exit(1)

    print(f"Found nunchaku at: {nunchaku_path}")

    # Check current status
    is_installed = check_chroma_installed(nunchaku_path)
    is_importable = verify_chroma_importable()

    print(f"Chroma files present: {is_installed}")
    print(f"Chroma importable: {is_importable}")

    if args.verify:
        if is_installed and is_importable:
            print("\nChroma support is correctly installed!")
            sys.exit(0)
        else:
            print("\nChroma support is NOT fully installed")
            sys.exit(1)

    # Perform patching
    if is_installed and is_importable:
        print("\nChroma support is already installed. Use --verify to check status.")
        if not args.comfyui:
            sys.exit(0)
    else:
        print("\nPatching nunchaku...")
        success = patch_nunchaku(nunchaku_path, dry_run=args.dry_run)

        if not success:
            print("Error: Failed to patch nunchaku")
            sys.exit(1)

        if not args.dry_run:
            # Verify
            # Need to reload the module
            import importlib
            import nunchaku
            importlib.reload(nunchaku)

            if verify_chroma_importable():
                print("\nSuccess! Chroma support is now installed.")
            else:
                print("\nWarning: Patching completed but import verification failed.")
                print("You may need to restart Python to use the new module.")

    # Patch ComfyUI if requested
    if args.comfyui:
        comfyui_path = Path(args.comfyui)
        if not comfyui_path.exists():
            print(f"\nError: ComfyUI-nunchaku path does not exist: {comfyui_path}")
            sys.exit(1)

        print()
        success = patch_comfyui_nunchaku(comfyui_path, dry_run=args.dry_run)

        if success:
            print("\nComfyUI-nunchaku patched successfully!")
        else:
            print("\nError: Failed to patch ComfyUI-nunchaku")
            sys.exit(1)

    if args.dry_run:
        print("\n(Dry run - no changes were made)")


if __name__ == "__main__":
    main()
