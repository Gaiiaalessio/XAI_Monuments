#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install torch
#pip install --upgrade tensorflow
#pip install --upgrade pip
# Carica il modulo CUDA 11.7
#!module load cuda/11.7
#!pip install shap
#!pip install numpy --upgrade
#!pip install numpy --upgrade pip
#!pip install pandas --upgrade
#!pip uninstall pandas -y
#!pip install pandas
#!pip install pycocotools
#pip install "numpy<2.0.0"


# In[2]:


##Imports
import numpy as np 
import torch
import os, sys, argparse, datetime, shutil
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tensorflow import keras
from tensorflow.python.keras.models import model_from_json
from keras.utils import to_categorical
import shap 
import tensorflow as tf
import torchvision
import time, json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.utils import to_categorical
tf.compat.v1.enable_eager_execution()

#Import from current folder
from utils.config import *
from utils.dataloader import *
from utils.engine import train_one_epoch, evaluate
from utils.train import compute_json_detection
from utils.knowledge_graph import compare_shap_and_KG, reduce_shap, GED_metric, get_bbox_weight
import utils.utils as uti


# In[3]:


import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# In[4]:


os.getcwd()


# In[5]:


os.environ["TF_XLA_FLAGS"] = ""


# In[6]:


#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda-11.7"
#os.environ["PATH"] = "/usr/local/cuda-11.7/bin:" + os.environ["PATH"]
#os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-11.7/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
#pip uninstall -y numpy tensorflow
#pip install numpy==1.24 tensorflow


# In[7]:


# Imposta il percorso della directory manualmente
os.chdir(os.path.expanduser('~/XAI_Monuments/tools'))
notebook_dir = os.getcwd()
sys.path.append(notebook_dir)
from pickle_tools import *
from metadata_tools import *
from monumai.monument import Monument

TMP_TRAIN = TMP_PATH + '/train'
TMP_VAL = TMP_PATH + '/val'
TMP_TEST = TMP_PATH + '/test'

os.makedirs(TMP_VAL, exist_ok=True)
os.makedirs(TMP_TRAIN, exist_ok=True)
os.makedirs(TMP_TEST, exist_ok=True)

# Argparse
parser = argparse.ArgumentParser(description='Arguments needed to prepare the metadata files')
parser.add_argument('--resume', dest='resume', help='Whether or not to resume a training', default=False)
parser.add_argument('--path_resume', dest='path_resume', help='Path to the model to load', default='./model/model_monumenai.pth')
parser.add_argument('--epoch_classif', dest='epoch_classif', help='Number of epochs to train the classification model', default=50)
parser.add_argument('--batch_size', dest='batch_size', help='Batch size to train the classification model', default=32)
parser.add_argument('--neuron_classif', dest='neuron_classif', help='Number of neurons in the classification model', default=11)
parser.add_argument('--epoch_detection', dest='epoch_detection', help='Number of epochs to train the detection model', default=0)
parser.add_argument('--lr', dest='lr', help='Learning rate of the detection model', default=0.0003)
parser.add_argument('--stepLR', dest='stepLR', help='Step of the learning rate scheduler', default=9)
parser.add_argument('--gammaLR', dest='gammaLR', help='Gamma parameter of the learning rate scheduler', default=0.1)
parser.add_argument('--weight', dest='weight', help='Type of weighting', default='None')
parser.add_argument('--exp_weights', dest='exp_weights', help='linear or exponential weighting', default='linear')
parser.add_argument('--data', dest='data', help='MonumenAI or PascalPart', default='MonumenAI')

# Se stai usando un ambiente Jupyter, ignora gli argomenti aggiuntivi
if 'ipykernel_launcher' in sys.argv[0]:
    sys.argv = sys.argv[:1]

args = parser.parse_args()


# In[8]:


#Hyperparameters classification
n_neurons_classification = int(args.neuron_classif)
num_epochs_classification = int(args.epoch_classif)
batch_size_classification = int(args.batch_size)
learning_rate_classification = None

#Hyperparameters detection
args.epoch_detection = 50
num_epochs_detection = int(args.epoch_detection)
learning_rate_detection = float(args.lr)
stepLR = float(args.stepLR)
gammaLR = float(args.gammaLR)
data = args.data


# In[9]:


if data == 'MonumenAI':
    archi_features = [el for sublist in list(Monument.ELEMENT_DIC.values()) for el in sublist]
    styles = FOLDERS_DATA
    #Loaders for detection
    PATH_DATA = "../"+PATH_DATA
    train_loader = ArchitectureDetectionDataset(os.path.join(PATH_DATA, 'train.csv'), os.path.join(PATH_DATA, CSV_XML), transform_detection)

    if num_epochs_detection != 0:
        val_loader = ArchitectureDetectionDataset(os.path.join(PATH_DATA,'val.csv'), os.path.join(PATH_DATA, CSV_XML), transform_detection)
    else:
        #Actually loading the test set
        val_loader = ArchitectureDetectionDataset(os.path.join(PATH_DATA,'val.csv'), os.path.join(PATH_DATA, CSV_XML), transform_detection)
        test_loader = ArchitectureDetectionDataset(os.path.join(PATH_DATA,'test.csv'), os.path.join(PATH_DATA, CSV_XML), transform_detection)


# In[10]:


##Hyperparameters for detection
num_archi_features = len(archi_features)
num_classes_detection = num_archi_features + 1  # num_archi_features + background
num_styles = len(styles)


# In[11]:


##Build detection model
if args.weight == "bbox_level":
    from utils.pytorch_utils import fasterrcnn_resnet50_fpn_custom
    detector = fasterrcnn_resnet50_fpn_custom(True)
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_detection)
else:
    detector = models.detection.fasterrcnn_resnet50_fpn(True)
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_detection)

if args.exp_weights == 'exponential':
    is_exponential = True
elif args.exp_weights == 'linear':
    is_exponential = False
else:
    print("Unrecognized type of weighting, defaulted to linear")
    is_exponential = False


# In[12]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
detector.to(device)

optimizer = torch.optim.SGD(detector.parameters(), lr=learning_rate_detection, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, stepLR, gamma=gammaLR, last_epoch=-1)

if args.resume:
    detector.load_state_dict(torch.load(args.path_resume, map_location=device))

# Classificatore
classificator = keras.Sequential([
    keras.layers.Dense(units=n_neurons_classification, activation='relu', input_shape=(num_archi_features,)),
    keras.layers.Dense(units=num_styles, activation='softmax')
])
classificator.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[13]:


def compute_json_detection(detector, loader, path, dataset = 'MonumenAI'):
    #Run inference on all data to prepare for classification
    detector.eval()
    
    if dataset == 'MonumenAI':
        information_about_class = ['M', 'G', 'R', 'B']
        elemen_dic = SUB_ELEMENTS
        reverse_dic = SUB_ELEMENTS_REVERSED
    
    for k in range(len(loader)):
        if k > 20: break
        percent = k/len(loader)*100
        print(f"Detection inference for `{path}`: {percent:.1f}%", end='\r')
        
        img = loader[k][0][0].cuda()
        results = detector([img])[0]
        r_b = results['boxes'].detach().cpu().numpy()
        scores = results['scores'].detach().cpu().numpy()
        classes = results['labels'].detach().cpu().numpy()
        unique, counts = np.unique(classes, return_counts=True)
        counter = dict(zip(unique, counts))

        if dataset == 'MonumenAI':
            img_name = loader.images_loc.iloc[k, 0]

        results = {}
        results["num_predictions"] = []
        results["image"] = img_name
        results["object"] = []
        
        if dataset == 'MonumenAI':
            results["true_label"] = int(loader.images_loc.iloc[k, 1])

        for name in elemen_dic:
            if elemen_dic[name] in counter:
                results["num_predictions"].append({
                    name :  int(counter[elemen_dic[name]])
                })
            else:
                results["num_predictions"].append({
                    name :  0
                })
            
        for k in range(len(r_b)):
            if classes[k] in reverse_dic:
                box = r_b[k]/224.
                local_result = {
                    "bndbox" : {
                        "xmin": str(box[0]),
                        "ymin": str(box[1]),
                        "ymax": str(box[3]),
                        "xmax": str(box[2])
                    },
                    "score" : str(scores[k]),
                    "class" : reverse_dic[classes[k]]
                }
                results["object"].append(local_result)
        
        local_path = os.path.join(path, information_about_class[results["true_label"]] + '_' + img_name.split('/')[-1][:-4] + '.json')
        
        with open(local_path, 'w') as fp:
            json.dump(results, fp)


# In[14]:


# Configurazione directory checkpoint
CHECKPOINT_DIR = "./checkpoints3/"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Funzione per caricare il checkpoint più recente
def load_checkpoint():
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("detector")]
    if checkpoints:
        last_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))
        checkpoint_path = os.path.join(CHECKPOINT_DIR, last_checkpoint)
        checkpoint = torch.load(checkpoint_path)
        print(f"Checkpoint caricato: {checkpoint_path}")
        return checkpoint
    return None

checkpoint = load_checkpoint()
start_epoch = checkpoint['epoch'] + 1 if checkpoint else 0

# Caricamento del modello e dello stato ottimizzatore
if checkpoint:
    detector.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# Funzione per preparare i dati per il classificatore
def prepare_data(loader, TMP, dataset):
    matrix_metadata = metadata_to_matrix(TMP, "json")
    names = matrix_metadata[:, -1]
    data = np.zeros((len(names), num_archi_features))
    labels = np.zeros(len(names))

    if dataset == "MonumenAI":
        for i, name in enumerate(names):
            im_name = name[2:-4]
            idx = loader.images_loc['path'].str.contains(im_name)
            data[idx] = matrix_metadata[i, :num_archi_features]
            labels[idx] = matrix_metadata[i, num_archi_features]
            
    data = data.astype(np.float32)
    labels = to_categorical(labels.astype(np.float32).astype(np.int8))
    return data, labels


# In[ ]:


from tqdm import tqdm  # Libreria per le barre di progresso
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.utils import to_categorical
from tensorflow import keras

# Configurazione directory checkpoint
CHECKPOINT_DIR = "./checkpoints3/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Funzione per caricare l'ultimo checkpoint salvato
def load_latest_checkpoints(detector, classificator, optimizer, scheduler):
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("detector_epoch")]
    if checkpoints:
        last_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))
        checkpoint_path = os.path.join(CHECKPOINT_DIR, last_checkpoint)
        print(f"Caricamento ultimo checkpoint del detector da: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)
        detector.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        print("Nessun checkpoint rilevato per il detector, avvio del training da zero.")
        start_epoch = 0

    classificator_checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("classificator_epoch")]
    if classificator_checkpoints:
        last_classificator_checkpoint = max(classificator_checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))
        classificator_path = os.path.join(CHECKPOINT_DIR, last_classificator_checkpoint)
        print(f"Caricamento ultimo checkpoint del classificatore da: {classificator_path}")
        classificator.load_weights(classificator_path)
    else:
        print("Nessun checkpoint del classificatore trovato, avvio del training da zero.")

    return start_epoch

# Funzione per calcolare la perdita SHAP
def calculate_shap_loss(shap_values, kg_matrix, threshold=0.05):
    loss = 0
    for i in range(shap_values.shape[0]):  # Loop su ogni immagine
        for j in range(shap_values.shape[1]):  # Loop su ogni caratteristica
            shap_value = shap_values[i, j]
            expected = kg_matrix[j]
            if abs(shap_value) > threshold:  # Caratteristica considerata rilevante
                if shap_value * expected < 0:  # Segno opposto a quanto previsto dal KG
                    loss += abs(shap_value)
    return loss / shap_values.shape[0]  # Normalizza la perdita

# Funzione per calcolare la perdita combinata
def compute_combined_loss(detector_loss, shap_loss, lambda_weight=0.1):
    return detector_loss + lambda_weight * shap_loss

# Funzione per preparare i dati per il classificatore
def prepare_data(loader, TMP, dataset):
    matrix_metadata = metadata_to_matrix(TMP, "json")
    names = matrix_metadata[:, -1]
    data = np.zeros((len(names), num_archi_features))
    labels = np.zeros(len(names))

    if dataset == "MonumenAI":
        for i, name in enumerate(names):
            im_name = name[2:-4]
            idx = loader.images_loc['path'].str.contains(im_name)
            data[idx] = matrix_metadata[i, :num_archi_features]
            labels[idx] = matrix_metadata[i, num_archi_features]
    elif dataset == "PascalPart":
        for i, name in enumerate(names):
            im_name = os.path.join(PATH_PASCAL + PASCAL_IMG, name.split('_')[1][:-5] + '.jpg')
            idx = loader.images_loc.index(im_name)
            data[idx] = matrix_metadata[i, :num_archi_features]
            labels[idx] = matrix_metadata[i, num_archi_features]

    data = data.astype(np.float32)
    labels = to_categorical(labels.astype(np.float32).astype(np.int8))
    return data, labels

# Definizione del classificatore
classificator = keras.Sequential([
    keras.layers.Dense(units=n_neurons_classification, activation='relu', input_shape=(num_archi_features,)),
    keras.layers.Dense(units=num_styles, activation='softmax')
])
classificator.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

start_epoch = load_latest_checkpoints(detector, classificator, optimizer, scheduler)

print("Inizio training del detector e classificatore...")
for epoch in range(start_epoch, num_epochs_detection):
    print(f"Epoch {epoch}/{num_epochs_detection}")

    # Training del detector
    detector.train()
    epoch_detector_loss = 0
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch} - Detector", unit="batch") as pbar_detector:
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to('cuda') for img in images]
            targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]

            loss_dict = detector(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_detector_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            pbar_detector.set_postfix(loss=f"{losses.item():.4f}")
            pbar_detector.update(1)

    scheduler.step()
    print(f"Detector completato. Loss media: {epoch_detector_loss / len(train_loader):.4f}")

    # Preparazione dei dati per il classificatore
    train_data, train_label = prepare_data(train_loader, TMP_TRAIN, data)
    
    # Calcolo e stampa della distribuzione delle classi
    unique, counts = np.unique(np.argmax(train_label, axis=1), return_counts=True)
    class_distribution = dict(zip(unique, counts))
    
    print("Distribuzione delle classi nel training set:")
    for class_id, count in class_distribution.items():
        print(f"Classe {class_id}: {count} campioni")


    # Training del classificatore
    epoch_classificator_loss = 0
    epoch_classificator_accuracy = 0
    with tqdm(total=len(train_data), desc=f"Epoch {epoch} - Classificator", unit="batch") as pbar_classificator:
        for batch_idx, (batch_data, batch_labels) in enumerate(zip(train_data, train_label)):
            batch_data = np.expand_dims(batch_data, axis=0)
            batch_labels = np.expand_dims(batch_labels, axis=0)

            history = classificator.train_on_batch(batch_data, batch_labels)
            batch_loss, batch_accuracy = history[0], history[1]
            epoch_classificator_loss += batch_loss
            epoch_classificator_accuracy += batch_accuracy

            pbar_classificator.set_postfix(loss=f"{batch_loss:.4f}", accuracy=f"{batch_accuracy:.4f}")
            pbar_classificator.update(1)

    print(f"Classificatore completato. Loss media: {epoch_classificator_loss / len(train_data):.4f}, "
          f"Accuracy media: {epoch_classificator_accuracy / len(train_data):.4f}")

    torch.save({
        'epoch': epoch,
        'model_state_dict': detector.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, os.path.join(CHECKPOINT_DIR, f"detector_epoch_{epoch}.pth"))
    classificator.save_weights(os.path.join(CHECKPOINT_DIR, f"classificator_epoch_{epoch}.weights.h5"))

print("Training completato.")

# Calcolo degli SHAP values rispetto alla predizione finale
print("Calcolo degli SHAP values...")
shap_explainer = shap.KernelExplainer(classificator.predict, train_data)
shap_values = shap_explainer.shap_values(test_data, nsamples=30)

# Analisi dei contributi SHAP
for i, shap_val in enumerate(shap_values):
    print(f"SHAP values per classe {i}: {shap_val[:5]}")  # Mostra i primi 5 contributi

# Carica il Knowledge Graph
print("Caricamento del Knowledge Graph...")
kg_matrix = load_knowledge_graph()

# Calcolo della perdita SHAP
print("Calcolo della perdita SHAP...")
shap_loss = calculate_shap_loss(shap_values, kg_matrix)
print(f"Perdita SHAP calcolata: {shap_loss:.4f}")

# Fine-tuning del detector con L_SHAP
print("Inizio fine-tuning del detector...")
for epoch in range(fine_tuning_epochs):
    detector.train()
    epoch_fine_tuning_loss = 0
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{fine_tuning_epochs} - Fine-Tuning Detector", unit="batch") as pbar:
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to('cuda') for img in images]
            targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]

            # Perdita standard del detector
            loss_dict = detector(images, targets)
            base_loss = sum(loss for loss in loss_dict.values())

            # Perdita totale
            total_loss = compute_combined_loss(base_loss, shap_loss)
            epoch_fine_tuning_loss += total_loss.item()

            # Backpropagazione e aggiornamento pesi
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Aggiorna la barra di avanzamento
            pbar.set_postfix(total_loss=f"{total_loss.item():.4f}")
            pbar.update(1)

    print(f"Fine-Tuning Epoch {epoch} completata. Loss media: {epoch_fine_tuning_loss / len(train_loader):.4f}")

# Valutazione finale
print("Valutazione finale del classificatore...")
loss, accuracy = classificator.evaluate(test_data, test_label, verbose=1)
print(f"Loss finale: {loss:.4f}, Accuracy finale: {accuracy:.4f}")

# Calcolo della metrica GED
print("Calcolo della metrica GED...")
d = GED_metric(test_data, shap_values, dataset=data)
print(f"SHAP GED: {d:.4f}")

# Visualizzazione matrice di confusione
print("Creazione della matrice di confusione...")
predict_test = classificator.predict(test_data)
prediction = np.argmax(predict_test, axis=1)
true_labels = np.argmax(test_label, axis=1)

cm = confusion_matrix(true_labels, prediction)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=STYLES_HOTONE_ENCODE.keys())
disp.plot(include_values=True, cmap='viridis')
plt.savefig('confmatDeLDECAS.png')
print("Matrice di confusione salvata.")


# In[ ]:


shutil.rmtree(TMP_PATH)

