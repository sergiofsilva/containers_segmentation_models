import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Força uso de CPU
os.environ['SM_FRAMEWORK'] = 'tf.keras'
from tensorflow import keras
import tensorflow as tf
import cv2
import segmentation_models as sm
import numpy as np
import matplotlib.pyplot as plt
import gc
import shutil


# PARAMETROS E SETAGENS
model_name = 'dengue_vf_1024_resnet50.h5'

BACKBONE = 'resnet50'
SIZE_INPUT_UNET = 1024  # Tamanho de entrada para o modelo UNet
BATCH_SIZE = 12
LR = 0.00001
EPOCHS = 8

ROOT_PATH = './'
base_path = './datasets'
models_path = os.path.join(ROOT_PATH, 'models')


#DATASET_DIR = os.path.join(base_path,'semantic_drone_datasys_v2/')
DATASET_DIR = os.path.join(base_path,'dengue_vf_1024/')


# Verifica se os caminhos existem
assert os.path.exists(DATASET_DIR), f"Dataset path not found: {DATASET_DIR}"
assert os.path.exists(os.path.join(DATASET_DIR, "labels_rgb.txt")), "labels_rgb.txt not found!"

print(f"Using dataset from: {DATASET_DIR}")


x_train_dir = os.path.join(DATASET_DIR, 'train')
y_train_dir = os.path.join(DATASET_DIR, 'trainannot')

x_valid_dir = os.path.join(DATASET_DIR, 'val')
y_valid_dir = os.path.join(DATASET_DIR, 'valannot')

x_test_dir = os.path.join(DATASET_DIR, 'test')
y_test_dir = os.path.join(DATASET_DIR, 'testannot')

def visualize(dir_to_save=None, save_name=None, **images):
    """Salva imagens em arquivo em vez de mostrar na tela"""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        if image.ndim == 3 and image.shape[-1] == 3:
            plt.imshow(image)
        else:
            plt.imshow(image, cmap='gray')
    
    # Gera nome do arquivo
    filename = save_name if save_name else "visualization_" + "_".join(images.keys())
    save_path = f"{dir_to_save}/{filename}.png"
    
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Visualização salva em: {save_path}")



# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    if x_max - x_min == 0:
        x_max = x_min + 1 # avoid division by zero
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

def bin_to_rgb(mask_bin, classes_list,labels_dict):
    img_height, img_width = mask_bin.shape[:2]
    mask_rgb = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    # Para cada classe, adiciona a cor RGB correspondente à máscara
    for i in range(n_classes-1):
        #print(f"nome da classe de indice [{i}] = {classes_list[i]}")
        rgb_color = labels_dict[classes_list[i]]['rgb']
        class_pixels = mask_bin[:, :, i] == 1
        mask_rgb[class_pixels] = rgb_color  # Aplicar cor RGB diretamente na máscara

    return mask_rgb


# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        labels_dict (dict): dictionary mapping class names to RGB colors
        classes (list): list of class names to extract from segmentation mask
        augmentation (albumentations.Compose): data transformation pipeline
        preprocessing (albumentations.Compose): data preprocessing pipeline
    """

    def __init__(
            self,
            images_dir,
            masks_dir = None,
            labels_dict=None,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}

        print(f"Dataset path >> images = {images_dir}, masks = {masks_dir}")

        self.masks_dir = masks_dir
        # Filtragem das imagens por extensões permitidas
        self.ids = [
            filename for filename in os.listdir(images_dir)
            if os.path.splitext(filename)[1].lower() in allowed_extensions
            and os.path.isfile(os.path.join(images_dir, filename))
        ]
        #self.ids = files_in_dir.sort()
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        if masks_dir is None:
            self.masks_fps = None
        else:
            self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # Map class names to corresponding RGB values
        self.class_colors = [labels_dict[cls]['rgb'] for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # Read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
        image = cv2.resize(image, (SIZE_INPUT_UNET, SIZE_INPUT_UNET))
        
        if self.masks_dir is None:
            return image, None, None
        else:
            mask_path = os.path.splitext(self.masks_fps[i])[0] + '.png'
            mask = cv2.imread(mask_path)  # Read mask in color mode (BGR)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  # Garante que a máscara esteja em RGB
            mask_rgb = cv2.resize(mask, (SIZE_INPUT_UNET, SIZE_INPUT_UNET), interpolation=cv2.INTER_NEAREST)
        
            # Convert mask to one-hot encoding based on RGB values
            masks = [np.all(mask_rgb == color, axis=-1) for color in self.class_colors]
            mask = np.stack(masks, axis=-1).astype('float')
        
            # Add background if mask is not binary
            if mask.shape[-1] != 1:
                background = 1 - mask.sum(axis=-1, keepdims=True)
                mask = np.concatenate((mask, background), axis=-1)
        
        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        return image, mask, mask_rgb
    
    
    def __len__(self):
        return len(self.ids)


class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


def parse_labelsfile_to_dict(filename):
    with open(filename, 'r') as file:
        content = file.readlines()

    result = {}
    for line in content:
        if line.startswith('#') or not line.strip():
            continue
        
        # Ignora a parte após "::" e divide a linha pelo primeiro ":"
        line = line.split("::")[0].strip()
        parts = line.split(":")
        
        if len(parts) >= 2:
            class_name = parts[0].strip()
            rgb = list(map(int, parts[1].split(",")))
            result[class_name] = {'rgb': rgb}
    return result


# Lets look at data we have

labels_dict = parse_labelsfile_to_dict(os.path.join(DATASET_DIR, 'labels_rgb.txt'))
print(labels_dict)
CLASSES=['caixa_d_agua_aberta',
         'caixa_d_agua_fechada',
         'piscina_coberta',
         'piscina_limpa',
         'piscina_suja'
         ]
dataset = Dataset(x_train_dir,
                  y_train_dir, 
                  labels_dict, 
                  classes=CLASSES)
    


preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
#n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES))
activation = 'sigmoid' if n_classes == 1 else 'softmax'


#create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

# define optomizer
optim = keras.optimizers.Adam(LR)


# USANDO TODOS OS PESOS DE CLASSE COMO 1
class_weights=np.ones(n_classes)


print('class_weights:', class_weights)


#dice_loss = sm.losses.DiceLoss(class_weights=class_weights)
dice_loss = sm.losses.DiceLoss(tf.convert_to_tensor(class_weights, dtype=tf.float32)) 

focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)


metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, total_loss, metrics)

# Training dataset
train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    labels_dict, 
    classes=CLASSES,
    augmentation=None,
    preprocessing=None,
)

# Validation dataset 
valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    labels_dict, 
    classes=CLASSES,
    augmentation=None,
    preprocessing=None,
)

train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False )


callbacks = [
    keras.callbacks.ModelCheckpoint(os.path.join(models_path, model_name), save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
]

print(f"Lendo pesos salvos de treinamento anterior")
model.load_weights(os.path.join(models_path, model_name))


print(f" ... pesos lidos com sucesso!")

history = model.fit(
    train_dataloader,
    steps_per_epoch=len(train_dataloader),
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=valid_dataloader,
    validation_steps=len(valid_dataloader),
)