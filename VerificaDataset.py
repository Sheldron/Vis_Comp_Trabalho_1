import os
from PIL import Image

#Verifica se todas as imagens são aceitas. Deve retornar 4217 imagens válidas.
def VerifyDataset():
    dataset_path = 'dataset'
    image_files = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
                try:
                    # Tente abrir para verificar integridade
                    img = Image.open(os.path.join(root, file))
                    image_files.append(os.path.join(root, file))
                except Exception as e:
                    print(f"Erro ao abrir {file}: {e}")

    print(f"Total de imagens válidas: {len(image_files)}")
    
VerifyDataset()
