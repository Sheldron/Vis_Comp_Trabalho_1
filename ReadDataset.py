import tensorflow as tf
import numpy as np
import os

# Função para ler e processar imagens
def ProcessPath(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])  # define shape explicitamente
    img = tf.image.resize(img, [256, 256])
    img = img / 255.0  # normaliza para 0-1
    return img, label

# Diretório raiz do dataset
dataset_path = 'dataset'

# Listar classes a partir dos nomes das subpastas - ordenadas para consistência
class_names = sorted(os.listdir(dataset_path))
print("Classes encontradas:", class_names)

image_paths = []
labels = []

# Percorrer cada classe e carregar os caminhos das imagens
for label, class_name in enumerate(class_names):
    class_dir = os.path.join(dataset_path, class_name)
    for fname in os.listdir(class_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(class_dir, fname))
            labels.append(label)

# Converter para arrays numpy
image_paths = np.array(image_paths)
labels = np.array(labels)

print(f"Total de imagens: {len(image_paths)}")

# Criar o Dataset do TensorFlow
path_ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
image_label_ds = path_ds.map(ProcessPath, num_parallel_calls=tf.data.AUTOTUNE)

# Configurações de desempenho
BATCH_SIZE = 32
image_label_ds = image_label_ds.shuffle(buffer_size=1000)
image_label_ds = image_label_ds.batch(BATCH_SIZE)
image_label_ds = image_label_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Pronto para usar em treinamento
for images, labels in image_label_ds.take(1):
    print(images.shape, labels.numpy())
