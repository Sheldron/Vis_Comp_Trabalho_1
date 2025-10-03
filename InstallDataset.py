import kagglehub

#Baixa o dataset diretamente do site Kaggle
def InstallDataset():
    # Download latest version
    path = kagglehub.dataset_download("gunavenkatdoddi/eye-diseases-classification")

    #Mostra o caminho no qual o dataset foi salvo
    print("Path to dataset files:", path)

InstallDataset()