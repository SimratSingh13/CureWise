# use_grpcio.py
import sys
import os

# Force add the site-packages to path
site_packages = r'C:\CureWise\venv\Lib\site-packages'
if site_packages not in sys.path:
    sys.path.insert(0, site_packages)

# Now try importing
try:
    import grpcio
    print(f"✅ Success! grpcio version: {grpcio.__version__}")
    
    # Test your app imports
    import firebase_admin
    from google.cloud import firestore
    print("✅ Firebase imports work!")
    
except ImportError as e:
    print(f"❌ Error: {e}")
    print("\nSite-packages contents:")
    os.system(f'dir {site_packages}')