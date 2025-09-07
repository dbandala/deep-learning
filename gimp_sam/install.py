#!/usr/bin/env python3

"""
GIMP SAM Plugin Installer
Automatically installs the SAM plugin to GIMP's plugin directory
"""

import os
import sys
import shutil
import platform
from pathlib import Path

def get_gimp_plugin_dir():
    """Get GIMP plugin directory based on OS"""
    system = platform.system()
    home = Path.home()
    
    if system == "Darwin":  # macOS
        return home / "Library/Application Support/GIMP/2.10/plug-ins"
    elif system == "Linux":
        return home / ".config/GIMP/2.10/plug-ins"
    elif system == "Windows":
        appdata = Path(os.environ.get("APPDATA", ""))
        return appdata / "GIMP/2.10/plug-ins"
    else:
        raise OSError(f"Unsupported operating system: {system}")

def install_plugin():
    """Install the SAM plugin to GIMP"""
    
    print("🔧 GIMP SAM Plugin Installer")
    print("=" * 40)
    
    # Get source and destination paths
    script_dir = Path(__file__).parent
    source_files = [
        script_dir / "sam_segmentation.py",
        script_dir / "sam_processor.py"
    ]
    
    # Check if source files exist
    for file_path in source_files:
        if not file_path.exists():
            print(f"❌ Source file not found: {file_path}")
            return False
    
    try:
        # Get GIMP plugin directory
        plugin_dir = get_gimp_plugin_dir()
        target_dir = plugin_dir / "sam_segmentation"
        
        print(f"📂 GIMP plugin directory: {plugin_dir}")
        print(f"📂 Target directory: {target_dir}")
        
        # Create GIMP plugin directory if it doesn't exist
        plugin_dir.mkdir(parents=True, exist_ok=True)
        
        # Create target plugin directory
        target_dir.mkdir(exist_ok=True)
        
        # Copy files
        for source_file in source_files:
            target_file = target_dir / source_file.name
            shutil.copy2(source_file, target_file)
            print(f"📄 Copied: {source_file.name}")
            
            # Make executable on Unix systems
            if platform.system() in ["Darwin", "Linux"]:
                os.chmod(target_file, 0o755)
                print(f"🔧 Made executable: {source_file.name}")
        
        print("\n✅ Plugin installed successfully!")
        print("\n📋 Next steps:")
        print("1. Restart GIMP")
        print("2. Open an image")
        print("3. Go to Filters → AI → SAM Segmentation...")
        print("4. Enjoy AI-powered image segmentation!")
        
        return True
        
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        return False

def uninstall_plugin():
    """Remove the SAM plugin from GIMP"""
    
    print("🗑️  GIMP SAM Plugin Uninstaller")
    print("=" * 40)
    
    try:
        plugin_dir = get_gimp_plugin_dir()
        target_dir = plugin_dir / "sam_segmentation"
        
        if target_dir.exists():
            shutil.rmtree(target_dir)
            print("✅ Plugin uninstalled successfully!")
        else:
            print("⚠️  Plugin directory not found - may already be uninstalled")
        
        return True
        
    except Exception as e:
        print(f"❌ Uninstallation failed: {e}")
        return False

def check_gimp_installation():
    """Check if GIMP is installed and accessible"""
    
    print("🔍 Checking GIMP installation...")
    
    plugin_dir = get_gimp_plugin_dir()
    
    if plugin_dir.parent.exists():
        print(f"✅ GIMP directory found: {plugin_dir.parent}")
        return True
    else:
        print(f"❌ GIMP directory not found: {plugin_dir.parent}")
        print("   Please make sure GIMP 2.10+ is installed")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Install/uninstall SAM plugin for GIMP")
    parser.add_argument("--uninstall", action="store_true", help="Uninstall the plugin")
    parser.add_argument("--check", action="store_true", help="Check GIMP installation only")
    
    args = parser.parse_args()
    
    if args.check:
        if check_gimp_installation():
            print("✅ GIMP is ready for plugin installation")
        sys.exit(0)
    
    if not check_gimp_installation():
        sys.exit(1)
    
    if args.uninstall:
        success = uninstall_plugin()
    else:
        success = install_plugin()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
