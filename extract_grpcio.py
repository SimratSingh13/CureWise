import zipfile 
with zipfile.ZipFile('grpcio-1.62.1-cp312-cp312-win_amd64.whl', 'r') as zip_ref: 
    zip_ref.extractall('venv/Lib/site-packages/') 
print('? Extracted grpcio successfully') 
