import cv2
import numpy as np
import os
import random

caminho_original = r'dataset'
caminho_aumentado = r'dataset_aumentado'

#Funções de Transformação 
def aplicar_rotacao(imagem, angulo=15):
    altura, largura = imagem.shape[:2]
    centro = (largura // 2, altura // 2)
    angulo_aleatorio = random.uniform(-angulo, angulo)
    matriz_rotacao = cv2.getRotationMatrix2D(centro, angulo_aleatorio, 1.0)
    imagem_rotacionada = cv2.warpAffine(imagem, matriz_rotacao, (largura, altura))
    return imagem_rotacionada

def aplicar_filtro_media(imagem, kernel_size=5):
    return cv2.blur(imagem, (kernel_size, kernel_size))


def aplicar_ruido_sal_pimenta(imagem, proporcao=0.01):
    """Adiciona ruído 'sal e pimenta' à imagem."""
    imagem_ruido = np.copy(imagem)
    altura, largura = imagem.shape[:2]
    num_pixels = int(altura * largura * proporcao)

    # Adicionar pimenta (pixels pretos)
    for _ in range(num_pixels // 2):
        y = random.randint(0, altura - 1)
        x = random.randint(0, largura - 1)
        imagem_ruido[y, x] = 0

    # Adicionar sal (pixels brancos)
    for _ in range(num_pixels // 2):
        y = random.randint(0, altura - 1)
        x = random.randint(0, largura - 1)
        imagem_ruido[y, x] = 255

    return imagem_ruido


# --- Lógica Principal ---
print(f"Verificando diretório de origem: {caminho_original}")
if not os.path.isdir(caminho_original):
    print("ERRO: O diretório de origem não foi encontrado!")
else:
    print("Diretório de origem OK.")

if not os.path.exists(caminho_aumentado):
    os.makedirs(caminho_aumentado)
    print(f"Diretório de destino criado em: {caminho_aumentado}")

classes = os.listdir(caminho_original)
print(f"Classes encontradas: {classes}")

for nome_classe in classes:
    dir_classe_original = os.path.join(caminho_original, nome_classe)
    dir_classe_aumentado = os.path.join(caminho_aumentado, nome_classe)

    if not os.path.exists(dir_classe_aumentado):
        os.makedirs(dir_classe_aumentado)

    nomes_arquivos = os.listdir(dir_classe_original)
    if not nomes_arquivos:
        print(f"AVISO: Nenhuma imagem encontrada na classe: {nome_classe}")
        continue

    amostra_arquivos = random.sample(nomes_arquivos, min(len(nomes_arquivos), 5))
    print(f"\nProcessando {len(amostra_arquivos)} imagens da classe: {nome_classe}")

    for nome_arquivo in amostra_arquivos:
        caminho_arquivo = os.path.join(dir_classe_original, nome_arquivo)
       
        print(f"  Tentando ler o arquivo: {caminho_arquivo}")
        
        # Solução para caminhos com caracteres especiais
        try:
            stream = open(caminho_arquivo, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            imagem = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        except Exception as e:
            print(f"    ERRO ao abrir o arquivo: {e}")
            imagem = None
        

        if imagem is not None:
            print("    Imagem lida com sucesso!")
            
            # Rotação
            img_rotacionada = aplicar_rotacao(imagem)
            nome_novo_arquivo_rot = f"rot_{nome_arquivo}"
            caminho_salvar_rot = os.path.join(dir_classe_aumentado, nome_novo_arquivo_rot)
            cv2.imwrite(caminho_salvar_rot, img_rotacionada)

            # Filtro de Média
            img_media = aplicar_filtro_media(imagem)
            nome_novo_arquivo_media = f"media_{nome_arquivo}"
            caminho_salvar_media = os.path.join(dir_classe_aumentado, nome_novo_arquivo_media)
            cv2.imwrite(caminho_salvar_media, img_media)

            # Filtro de Ruído
            img_ruido = aplicar_ruido_sal_pimenta(imagem)
            nome_novo_arquivo_ruido = f"ruido_{nome_arquivo}"
            caminho_salvar_ruido = os.path.join(dir_classe_aumentado, nome_novo_arquivo_ruido)
            cv2.imwrite(caminho_salvar_ruido, img_ruido)

    
        else:
            # Se a imagem for None, nos avise!
            print("    FALHA AO LER IMAGEM. O arquivo pode estar corrompido ou o caminho está incorreto.")

print("\nProcesso de aumento de dados (teste) concluído!")
