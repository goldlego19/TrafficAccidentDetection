"""
Diagnostic script to check dataset structure
"""

from pathlib import Path

def diagnose_tad_dataset(root_dir='TAD/frames'):
    """Check what's in the TAD dataset"""
    root = Path(root_dir)
    
    print(f"Diagnosing TAD dataset at: {root. absolute()}")
    print("=" * 80)
    
    # Check abnormal directory
    abnormal_dir = root / 'abnormal'
    if abnormal_dir. exists():
        print(f"\n📁 ABNORMAL directory:  {abnormal_dir}")
        items = list(abnormal_dir. iterdir())
        print(f"   Total items: {len(items)}")
        
        for i, item in enumerate(items[:5]):  # Show first 5
            print(f"\n   [{i+1}] {item.name}")
            print(f"       Is directory: {item.is_dir()}")
            
            if item.is_dir():
                # List image files
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
                all_images = []
                for ext in image_extensions:
                    all_images.extend(list(item. glob(ext)))
                
                print(f"       Image files: {len(all_images)}")
                if len(all_images) > 0:
                    print(f"       Sample:  {all_images[0].name}")
                else:
                    # Show what IS in there
                    contents = list(item.iterdir())[: 5]
                    print(f"       Contents ({len(list(item.iterdir()))} total):")
                    for c in contents:
                        print(f"         - {c.name} ({'dir' if c.is_dir() else 'file'})")
    
    # Check normal directory
    normal_dir = root / 'normal'
    if normal_dir.exists():
        print(f"\n📁 NORMAL directory: {normal_dir}")
        items = list(normal_dir.iterdir())
        print(f"   Total items:  {len(items)}")
        
        # Find the problematic directory
        problem_dir = normal_dir / 'Normal_174.mp4'
        if problem_dir.exists():
            print(f"\n   ⚠️  FOUND PROBLEMATIC DIRECTORY: {problem_dir}")
            print(f"       Is directory: {problem_dir. is_dir()}")
            
            # List all files with all extensions
            all_files = list(problem_dir.iterdir())
            print(f"       Total files: {len(all_files)}")
            
            if len(all_files) > 0:
                print(f"       Files in directory:")
                for f in all_files[: 10]:  # Show first 10
                    print(f"         - {f.name} ({f.suffix})")
            
            # Try specific extensions
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*. PNG', '*.BMP']
            for ext in image_extensions:
                matches = list(problem_dir.glob(ext))
                if matches:
                    print(f"       Found {len(matches)} files matching {ext}")
        
        # Show other normal directories
        print(f"\n   Other normal directories (showing first 5):")
        for i, item in enumerate(items[: 5]):
            print(f"       [{i+1}] {item.name}")
            if item.is_dir():
                image_files = list(item.glob('*. jpg')) + list(item.glob('*. png'))
                print(f"            Images: {len(image_files)}")


if __name__ == '__main__':
    diagnose_tad_dataset()