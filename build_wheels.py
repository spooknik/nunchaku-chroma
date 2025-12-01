#!/usr/bin/env python3
"""
Nunchaku-Chroma Wheel Repackaging Script

This script downloads official nunchaku wheels from HuggingFace, injects Chroma
model support files, updates the package metadata, and creates new wheels with
the name 'nunchaku-chroma'.

Usage:
    python build_wheels.py                    # Build all wheels for latest version
    python build_wheels.py --version 1.0.2    # Build specific version
    python build_wheels.py --single torch2.6 cp312 linux_x86_64  # Build single wheel
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configuration
NUNCHAKU_VERSIONS_URL = "https://nunchaku.tech/cdn/nunchaku_versions.json"
HUGGINGFACE_BASE_URL = "https://huggingface.co/nunchaku-tech/nunchaku/resolve/main"

SUPPORTED_PLATFORMS = ["linux_x86_64", "win_amd64"]
SUPPORTED_PYTHON = ["cp310", "cp311", "cp312", "cp313"]
# Updated to match what's available for nunchaku 1.0.2
SUPPORTED_TORCH = ["torch2.7", "torch2.8", "torch2.9", "torch2.10"]

# Paths
SCRIPT_DIR = Path(__file__).parent
CHROMA_FILES_DIR = SCRIPT_DIR / "chroma_files"
DIST_DIR = SCRIPT_DIR / "dist"


def fetch_nunchaku_versions() -> Dict:
    """Fetch the nunchaku version manifest from CDN."""
    print(f"Fetching version info from {NUNCHAKU_VERSIONS_URL}...")
    try:
        req = urllib.request.Request(
            NUNCHAKU_VERSIONS_URL,
            headers={"User-Agent": "nunchaku-chroma-builder"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read())
    except Exception as e:
        raise RuntimeError(f"Failed to fetch version info: {e}")


def get_latest_version(versions_info: Dict) -> str:
    """Get the latest stable version from the version manifest."""
    versions = versions_info.get("versions", [])
    if not versions:
        raise RuntimeError("No versions found in manifest")
    return versions[0]  # First is latest


def construct_wheel_url(version: str, torch_ver: str, py_ver: str, platform: str) -> str:
    """Construct the HuggingFace download URL for a wheel."""
    filename = f"nunchaku-{version}+{torch_ver}-{py_ver}-{py_ver}-{platform}.whl"
    return f"{HUGGINGFACE_BASE_URL}/{filename}", filename


def download_wheel(url: str, dest_path: Path) -> None:
    """Download a wheel file from URL."""
    print(f"  Downloading from {url}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "nunchaku-chroma-builder"})
        with urllib.request.urlopen(req, timeout=300) as response:
            with open(dest_path, "wb") as f:
                shutil.copyfileobj(response, f)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise FileNotFoundError(f"Wheel not found: {url}")
        raise


def extract_wheel(wheel_path: Path, dest_dir: Path) -> None:
    """Extract a wheel (zip) file to destination directory."""
    with zipfile.ZipFile(wheel_path, "r") as zf:
        zf.extractall(dest_dir)


def inject_chroma_files(package_dir: Path) -> None:
    """Copy Chroma files into the extracted nunchaku package."""
    nunchaku_dir = package_dir / "nunchaku"

    # Copy transformer_chroma.py
    src_transformer = CHROMA_FILES_DIR / "models" / "transformers" / "transformer_chroma.py"
    dst_transformer = nunchaku_dir / "models" / "transformers" / "transformer_chroma.py"
    if src_transformer.exists():
        shutil.copy2(src_transformer, dst_transformer)
        print(f"    Copied transformer_chroma.py")
    else:
        raise FileNotFoundError(f"Missing: {src_transformer}")

    # Copy lora/chroma directory
    src_lora = CHROMA_FILES_DIR / "lora" / "chroma"
    dst_lora = nunchaku_dir / "lora" / "chroma"
    if src_lora.exists():
        if dst_lora.exists():
            shutil.rmtree(dst_lora)
        shutil.copytree(src_lora, dst_lora)
        print(f"    Copied lora/chroma/")
    else:
        raise FileNotFoundError(f"Missing: {src_lora}")


def patch_init_files(package_dir: Path) -> None:
    """Patch __init__.py files to export Chroma classes."""
    nunchaku_dir = package_dir / "nunchaku"

    # Patch nunchaku/models/transformers/__init__.py
    transformers_init = nunchaku_dir / "models" / "transformers" / "__init__.py"
    patch_init_file(
        transformers_init,
        "from .transformer_chroma import NunchakuChromaTransformer2DModel",
        "__all__ export for Chroma"
    )

    # Patch nunchaku/models/__init__.py
    models_init = nunchaku_dir / "models" / "__init__.py"
    patch_init_file(
        models_init,
        "from .transformers.transformer_chroma import NunchakuChromaTransformer2DModel",
        "models __init__ Chroma export"
    )

    # Patch nunchaku/__init__.py
    nunchaku_init = nunchaku_dir / "__init__.py"
    patch_init_file(
        nunchaku_init,
        "from .models.transformers.transformer_chroma import NunchakuChromaTransformer2DModel",
        "top-level Chroma export"
    )

    # Patch nunchaku/lora/__init__.py to include chroma submodule
    lora_init = nunchaku_dir / "lora" / "__init__.py"
    if lora_init.exists():
        patch_init_file(
            lora_init,
            "from . import chroma",
            "lora chroma submodule import"
        )


def patch_init_file(init_path: Path, import_line: str, description: str) -> None:
    """Add an import line to an __init__.py file if not already present."""
    if not init_path.exists():
        print(f"    Warning: {init_path} does not exist, creating it")
        init_path.write_text(f"{import_line}\n")
        return

    content = init_path.read_text()
    if import_line in content:
        print(f"    {description}: already patched")
        return

    # Add import at the end
    if not content.endswith("\n"):
        content += "\n"
    content += f"\n# Chroma model support\n{import_line}\n"
    init_path.write_text(content)
    print(f"    {description}: patched")


def update_metadata(dist_info_dir: Path, new_name: str = "nunchaku-chroma") -> None:
    """Update the METADATA file to change the package name."""
    metadata_path = dist_info_dir / "METADATA"
    if not metadata_path.exists():
        raise FileNotFoundError(f"METADATA not found: {metadata_path}")

    content = metadata_path.read_text()
    lines = content.split("\n")
    new_lines = []

    for line in lines:
        if line.startswith("Name: "):
            new_lines.append(f"Name: {new_name}")
        else:
            new_lines.append(line)

    metadata_path.write_text("\n".join(new_lines))
    print(f"    Updated METADATA: Name -> {new_name}")


def compute_file_hash(file_path: Path) -> Tuple[str, int]:
    """Compute SHA256 hash and size for a file (for RECORD)."""
    sha256 = hashlib.sha256()
    size = 0
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
            size += len(chunk)
    # Base64url encoding without padding
    import base64
    hash_b64 = base64.urlsafe_b64encode(sha256.digest()).rstrip(b"=").decode("ascii")
    return f"sha256={hash_b64}", size


def regenerate_record(package_dir: Path, dist_info_dir: Path) -> None:
    """Regenerate the RECORD file with updated checksums."""
    record_path = dist_info_dir / "RECORD"
    records = []

    # Walk through all files in the package
    for root, dirs, files in os.walk(package_dir):
        for filename in files:
            file_path = Path(root) / filename
            rel_path = file_path.relative_to(package_dir)

            # RECORD itself has no hash
            if rel_path == dist_info_dir.relative_to(package_dir) / "RECORD":
                records.append(f"{rel_path},,")
            else:
                hash_str, size = compute_file_hash(file_path)
                records.append(f"{rel_path},{hash_str},{size}")

    record_path.write_text("\n".join(records) + "\n")
    print(f"    Regenerated RECORD with {len(records)} entries")


def repackage_wheel(package_dir: Path, output_path: Path) -> None:
    """Repackage the modified files into a new wheel."""
    # Wheel is just a zip with .whl extension
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(package_dir):
            for filename in files:
                file_path = Path(root) / filename
                arc_name = file_path.relative_to(package_dir)
                zf.write(file_path, arc_name)

    print(f"    Created wheel: {output_path.name}")


def build_single_wheel(
    version: str,
    torch_ver: str,
    py_ver: str,
    platform: str,
    output_dir: Path
) -> Optional[Path]:
    """Build a single repackaged wheel."""
    print(f"\nBuilding: nunchaku-chroma-{version}+{torch_ver}-{py_ver}-{platform}")

    # Construct URLs and filenames
    url, orig_filename = construct_wheel_url(version, torch_ver, py_ver, platform)
    new_filename = f"nunchaku_chroma-{version}+{torch_ver}-{py_ver}-{py_ver}-{platform}.whl"
    output_path = output_dir / new_filename

    # Skip if already exists
    if output_path.exists():
        print(f"  Already exists, skipping")
        return output_path

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        wheel_path = tmpdir / orig_filename
        extract_dir = tmpdir / "extracted"
        extract_dir.mkdir()

        # Download
        try:
            download_wheel(url, wheel_path)
        except FileNotFoundError:
            print(f"  Wheel not available, skipping")
            return None

        # Extract
        print(f"  Extracting...")
        extract_wheel(wheel_path, extract_dir)

        # Find dist-info directory
        dist_info_dirs = list(extract_dir.glob("*.dist-info"))
        if not dist_info_dirs:
            raise RuntimeError("No dist-info directory found in wheel")
        dist_info_dir = dist_info_dirs[0]

        # Keep original dist-info name (nunchaku) so it's a drop-in replacement
        # This ensures ComfyUI-nunchaku can detect the package

        # Inject Chroma files
        print(f"  Injecting Chroma files...")
        inject_chroma_files(extract_dir)

        # Patch init files
        print(f"  Patching __init__.py files...")
        patch_init_files(extract_dir)

        # Keep original metadata (nunchaku) - don't rename to nunchaku-chroma
        # This makes it a true drop-in replacement

        # Regenerate RECORD
        print(f"  Regenerating RECORD...")
        regenerate_record(extract_dir, dist_info_dir)

        # Repackage
        print(f"  Repackaging...")
        repackage_wheel(extract_dir, output_path)

    return output_path


def build_all_wheels(version: str, output_dir: Path) -> List[Path]:
    """Build wheels for all supported combinations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    built_wheels = []

    total = len(SUPPORTED_PLATFORMS) * len(SUPPORTED_PYTHON) * len(SUPPORTED_TORCH)
    current = 0

    for platform in SUPPORTED_PLATFORMS:
        for py_ver in SUPPORTED_PYTHON:
            for torch_ver in SUPPORTED_TORCH:
                current += 1
                print(f"\n[{current}/{total}]", end="")

                result = build_single_wheel(version, torch_ver, py_ver, platform, output_dir)
                if result:
                    built_wheels.append(result)

    return built_wheels


def generate_versions_json(version: str, output_dir: Path) -> None:
    """Generate the nunchaku_chroma_versions.json manifest."""
    manifest = {
        "versions": [version],
        "supported_torch": SUPPORTED_TORCH,
        "supported_python": SUPPORTED_PYTHON,
        "filename_template": "nunchaku_chroma-{version}+{torch_version}-{python_version}-{python_version}-{platform}.whl",
        "url_templates": {
            "github": "https://github.com/spooknik/nunchaku-chroma/releases/download/v{version_tag}/{filename}"
        }
    }

    manifest_path = output_dir / "nunchaku_chroma_versions.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nGenerated: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Build nunchaku-chroma wheels")
    parser.add_argument("--version", help="Nunchaku version to repackage (default: latest)")
    parser.add_argument("--single", nargs=3, metavar=("TORCH", "PYTHON", "PLATFORM"),
                       help="Build single wheel (e.g., torch2.6 cp312 linux_x86_64)")
    parser.add_argument("--output", type=Path, default=DIST_DIR, help="Output directory")
    args = parser.parse_args()

    # Verify chroma_files directory exists
    if not CHROMA_FILES_DIR.exists():
        print(f"Error: {CHROMA_FILES_DIR} does not exist!")
        print("Please run setup_chroma_files.py first or create the directory structure manually.")
        return 1

    # Get version
    if args.version:
        version = args.version
    else:
        versions_info = fetch_nunchaku_versions()
        version = get_latest_version(versions_info)
        print(f"Latest version: {version}")

    # Build wheels
    args.output.mkdir(parents=True, exist_ok=True)

    if args.single:
        torch_ver, py_ver, platform = args.single
        result = build_single_wheel(version, torch_ver, py_ver, platform, args.output)
        if result:
            print(f"\nSuccess! Built: {result}")
        else:
            print(f"\nFailed to build wheel")
            return 1
    else:
        built = build_all_wheels(version, args.output)
        print(f"\n{'='*60}")
        print(f"Built {len(built)} wheels in {args.output}")

        # Generate versions manifest
        generate_versions_json(version, args.output)

    return 0


if __name__ == "__main__":
    exit(main())
