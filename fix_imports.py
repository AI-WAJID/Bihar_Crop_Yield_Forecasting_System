
import os
import re

def fix_imports_in_file(filepath, fixes):
    """Fix import statements in a file"""
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        for old_import, new_import in fixes.items():
            content = content.replace(old_import, new_import)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed imports in: {filepath}")
            return True
        else:
            print(f"ℹ️  No changes needed in: {filepath}")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing {filepath}: {e}")
        return False

def main():
    """Fix all import issues"""
    
    print("🔧 FIXING IMPORT STATEMENTS IN BIHAR CROP FORECASTING PROJECT")
    print("=" * 60)
    
    # Fix preprocessing.py imports
    preprocessing_fixes = {
        'from ingestion import DataIngestion': 'from src.data.ingestion import DataIngestion'
    }
    
    files_to_fix = [
        ('src/data/preprocessing.py', preprocessing_fixes),
    ]
    
    fixes_made = 0
    
    for filepath, fixes in files_to_fix:
        if os.path.exists(filepath):
            if fix_imports_in_file(filepath, fixes):
                fixes_made += 1
        else:
            print(f"⚠️  File not found: {filepath}")
    
    # Also check and fix any other potential import issues
    # Look for other files that might have similar issues
    other_files = [
        'src/models/train_model.py',
        'src/deployment/api.py'
    ]
    
    for filepath in other_files:
        if os.path.exists(filepath):
            # Check if this file has any relative imports that need fixing
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for common import patterns that need fixing
                patterns_to_fix = {
                    'from ingestion import': 'from src.data.ingestion import',
                    'from preprocessing import': 'from src.data.preprocessing import',
                    'from train_model import': 'from src.models.train_model import',
                }
                
                original_content = content
                for old_pattern, new_pattern in patterns_to_fix.items():
                    if old_pattern in content:
                        content = content.replace(old_pattern, new_pattern)
                
                if content != original_content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"✅ Fixed imports in: {filepath}")
                    fixes_made += 1
                    
            except Exception as e:
                print(f"❌ Error checking {filepath}: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 IMPORT FIXES SUMMARY")
    print("=" * 60)
    print(f"Files fixed: {fixes_made}")
    
    if fixes_made > 0:
        print("\n🎉 Import statements have been fixed!")
        print("\n🚀 Now you can run the project:")
        print("   python run_project.py")
    else:
        print("\n⚠️  No import issues found or unable to fix them.")
        print("You may need to manually check the import statements.")
    
    return fixes_made > 0

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Ready to run the project!")
    else:
        print("\n❌ Please check the import statements manually.")