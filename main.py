
import os
from collections import defaultdict
import math
import networkx as nx
import random
from tqdm import tqdm
from zipfile import ZipFile
from urllib.request import urlretrieve
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from sklearn.model_selection import KFold
from node2vec_model import create_model, create_dataset, generate_examples, next_step, random_walk
from sequence_coding import label_smiles, label_sequence
from contrastive_loss_functions import max_margin_contrastive_loss, multiclass_npairs_loss, triplet_loss, supervised_nt_xent_loss


data_path = '/content/drive/My Drive/Project/Project/DeepDTI/MultiDTI/data/'  # please set this line properly

# Read drug-drug matrix and Build drug_drug graph
drug_drug_graph = nx.Graph()
data = pd.read_csv(data_path + 'drug_drug.csv')

for i in range(1,len(data)):
  col=data[data.columns[i]]
  col=np.array(col)
  idx=np.argwhere(col)
  if idx.shape[0] != 0:
    #print(idx.shape[0])
    for j in range(0,idx.shape[0]):
      drug_drug_graph.add_edge(i-1, idx[j,0], weight=1)
  else:
    col=data[data.columns[i]]
    col=np.array(col)
    idx=np.argwhere(col==0)
    #print(idx.shape[0])
    for j in range(0,idx.shape[0]):
      drug_drug_graph.add_edge(i-1, idx[j,0], weight=0.0001)

# Read protein_protein matrix and Build protein_protein graph
protein_protein_graph = nx.Graph()

data = pd.read_csv(data_path + 'protein_protein.csv')

for i in range(1,1513):
  col=data[data.columns[i]]
  col=np.array(col)
  idx=np.argwhere(col==1)
  if idx.shape[0] != 0:
    col=data[data.columns[i]]
    col=np.array(col)
    idx=np.argwhere(col==1)
    #print(idx.shape[0])
    for j in range(0,idx.shape[0]):
      protein_protein_graph.add_edge(i-1, idx[j,0], weight=1)
  else:
    col=data[data.columns[i]]
    col=np.array(col)
    idx=np.argwhere(col==0)
    #print(idx.shape[0])
    for j in range(0,idx.shape[0]):
      protein_protein_graph.add_edge(i-1, idx[j,0], weight=0.0001)



## Read drug smiles

CHARCANSMISET = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6, 
			 ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12, 
			 "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18, 
			 "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
			 "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30, 
			 "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36, 
			 "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42, 
			 "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48, 
			 "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54, 
			 "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
			 "t": 61, "y": 62, "@": 63, "/": 64, "\\": 0}

smiles_dict_len = 64
smiles_max_len=100


smiles_samples = pd.read_csv(data_path + 'durg_smiles.csv',header=None)
smiles_rep=[]
for a in smiles_samples[1]:
   smiles_rep.append(label_smiles(a,smiles_max_len,CHARCANSMISET))


## Read protein sequence
CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 }

CHARPROTLEN = 25
protein_dict_len=65
protein_max_len=1000
protein_samples = pd.read_csv(data_path + 'protein_fasta.csv',header=None)
protein_rep=[]
for a in protein_samples[1]:
   protein_rep.append(label_sequence(a,protein_max_len,CHARPROTSET))

drug_protein_interaction = pd.read_csv(data_path + 'drug_protein.csv',header=0)
print(drug_protein_interaction)
print(drug_protein_interaction.iloc[0,0])

#####################################################################
# node2vec for Drug-Drug and Protein-Protein                        #
#####################################################################

count_p=0
count_n=0

positive_samples_d=[]
negative_samples_d=[]

positive_samples_p=[]
negative_samples_p=[]
for i in range(0,708):
  for j in range(1,1513):
    if drug_protein_interaction.iloc[i,j]==1:
       count_p=count_p+1
       positive_samples_d.append([i])
       positive_samples_p.append([j-1])
    else:
      count_n=count_n+1
      negative_samples_d.append([i])
      negative_samples_p.append([j-1])
print(count_p)
print(count_n)

idx = np.random.choice(count_n, count_p*10, replace=False)  
idx=np.reshape(idx,(-1,1))

positive_samples_d_orig=np.array(positive_samples_d)
negative_samples_d_orig=np.reshape(np.array(negative_samples_d)[idx],(-1,1))
positive_samples_p_orig=np.array(positive_samples_p)
negative_samples_p_orig=np.reshape(np.array(negative_samples_p)[idx],(-1,1))

aa=np.reshape(np.tile(positive_samples_d_orig,10),(19230,1))
bb=np.reshape(np.tile(positive_samples_p_orig,10),(19230,1))
c=list(zip(aa,bb))
random.shuffle(c)
a, b = zip(*c)
positive_samples_d_orig=np.reshape(a,(19230,1))
positive_samples_p_orig=np.reshape(b,(19230,1))

degrees = []
for node in drug_drug_graph.nodes:
    degrees.append(protein_protein_graph.degree[node])
print("Average node degree:", round(sum(degrees) / len(degrees), 2))



vocabulary_drug = ["NA"] + list(drug_drug_graph.nodes)
vocabulary_lookup_drug = {token: idx for idx, token in enumerate(vocabulary_drug)}

vocabulary_protein = ["NA"] + list(protein_protein_graph.nodes)
vocabulary_lookup_protein = {token: idx for idx, token in enumerate(vocabulary_protein)}

##############################################

# Random walk return parameter.
p = 1
# Random walk in-out parameter.
q = 1
# Number of iterations of random walks.
num_walks = 5
# Number of steps of each random walk.
num_steps = 10


walks_drug = random_walk(drug_drug_graph, num_walks,vocabulary_lookup_drug, num_steps, p, q)
walks_protein = random_walk(protein_protein_graph, num_walks,vocabulary_lookup_protein, num_steps, p, q)

print("Number of drug walks generated:", len(walks_drug))
print("Number of protein walks generated:", len(walks_protein))

#################################################

num_negative_samples = 10#4

# generate examples for Drug-Drug Network

targets_d, contexts_d, labels_d, weights_d = generate_examples(
    sequences=walks_drug,
    window_size=num_steps,
    num_negative_samples=num_negative_samples,
    vocabulary_size=len(vocabulary_drug),
)

# generate examples for Protein-Protein Network

targets_p, contexts_p, labels_p, weights_p = generate_examples(
    sequences=walks_protein,
    window_size=num_steps,
    num_negative_samples=num_negative_samples,
    vocabulary_size=len(vocabulary_protein),
)



batch_size = 1024


dataset_drug = create_dataset(
    targets=targets_d,
    contexts=contexts_d,
    labels=labels_d,
    weights=weights_d,
    batch_size=batch_size,
)

dataset_protein = create_dataset(
    targets=targets_p,
    contexts=contexts_p,
    labels=labels_p,
    weights=weights_p,
    batch_size=batch_size,
)


learning_rate = 0.001


drug_embeding_dim=100
protein_embeding_dim=200

model_drug = create_model(len(vocabulary_drug), drug_embeding_dim)
model_drug.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
)

model_protein = create_model(len(vocabulary_protein), protein_embeding_dim)
model_protein.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
)

# 
history_drug = model_drug.fit(dataset_drug, epochs=100)
history_protein = model_protein.fit(dataset_protein, epochs=100)
drug_embeddings = model_drug.get_layer("item_embeddings").get_weights()[0]
print("Drug Embeddings shape:", drug_embeddings.shape)
protein_embeddings = model_protein.get_layer("item_embeddings").get_weights()[0]
print("Protein Embeddings shape:", protein_embeddings.shape)

############################################################
#               Proposed deep learing based DTI model      #
############################################################
###   Define model architecture for drug and protein
import tensorflow
import keras
from keras.models import Model
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import Conv2D, GRU
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Masking, RepeatVector, Flatten, Concatenate
from keras.models import Model
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers, layers
from tensorflow.keras import regularizers

embedding_size=256
num_filters=64
protein_filter_lengths=8
smiles_filter_lengths=4

Drug_input = Input(shape=(smiles_max_len,), dtype='int32',name='drug_input')
Protein_input = Input(shape=(protein_max_len,), dtype='int32',name='protein_input')

Drug_Drug_input = Input(shape=(drug_embeding_dim,),name='drug_drug_input')
Protein_Protein_input = Input(shape=(protein_embeding_dim,),name='protein_protein_input')

encode_smiles = Embedding(input_dim=smiles_dict_len+1, output_dim = embedding_size, input_length=smiles_max_len,name='smiles_embedding')(Drug_input)
encode_smiles = Conv1D(filters=num_filters, kernel_size=smiles_filter_lengths,  activation='relu', padding='valid',  strides=1, name='conv1_smiles')(encode_smiles)
# encode_smiles = MaxPooling1D(2)(encode_smiles)  # test it is effectuve or not
encode_smiles = Conv1D(filters=num_filters*2, kernel_size=smiles_filter_lengths,  activation='relu', padding='valid',  strides=1, name='conv2_smiles')(encode_smiles)
#encode_smiles = MaxPooling1D(2)(encode_smiles)
encode_smiles = Conv1D(filters=num_filters*3, kernel_size=smiles_filter_lengths,  activation='relu', padding='valid',  strides=1, name='conv3_smiles')(encode_smiles)
#encode_smiles = MaxPooling1D(2)(encode_smiles)


encode_protein = Embedding(input_dim=protein_dict_len+1, output_dim = embedding_size, input_length=protein_max_len, name='protein_embedding')(Protein_input)
encode_protein = Conv1D(filters=num_filters, kernel_size=protein_filter_lengths,  activation='relu', padding='valid',  strides=1, name='conv1_prot')(encode_protein)
#encode_protein = MaxPooling1D(2)(encode_protein)
encode_protein = Conv1D(filters=num_filters*2, kernel_size=protein_filter_lengths,  activation='relu', padding='valid',  strides=1, name='conv2_prot')(encode_protein)
#encode_protein = MaxPooling1D(2)(encode_protein)
encode_protein = Conv1D(filters=num_filters*3, kernel_size=protein_filter_lengths,  activation='relu', padding='valid',  strides=1, name='conv3_prot')(encode_protein)
#encode_protein = MaxPooling1D(2)(encode_protein)

encode_protein = GlobalMaxPooling1D()(encode_protein)
encode_smiles = GlobalMaxPooling1D()(encode_smiles)

encode_interaction =Concatenate()([encode_protein,encode_smiles])
encode_interaction2 =Concatenate()([Protein_Protein_input,Drug_Drug_input])
feat_vec = Concatenate()([encode_interaction, encode_interaction2])
#encode_interaction =Concatenate()([encode_interaction,encode_interaction2])

# Fully connected
FC1 = Dense(1024, activation='relu', name='dense1')(encode_interaction)
FC2 = Dropout(0.5)(FC1)
FC2 = Dense(1024, activation='relu', name='dense2')(FC2)
FC2 = Dropout(0.5)(FC2)
FC2 = Dense(512, activation='relu', name='dense3')(FC2) #'dense3'

FC1_2 = Dense(1024, activation='relu', name='dense1_')(encode_interaction2)
FC2_2 = Dropout(0.5)(FC1_2)
FC2_2 = Dense(1024, activation='relu', name='dense2_')(FC2_2)
FC2_2 = Dropout(0.5)(FC2_2)
FC2_2 = Dense(512, activation='relu', name='dense3_')(FC2_2)

FC2 =Concatenate(name='output1')([FC2,FC2_2])

FC3 = Dense(512, activation='relu', name='dense3_p')(FC2)
#FC2 = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1),name='output1')(FC2)
# And add a logistic regression on top
predictions = Dense(1, activation='sigmoid', name='dense4')(FC3) # if you want train model for active/inactive set activation='sigmoid'
#,activity_regularizer=regularizers.l2(1e-5)
embedding_model = Model(inputs=[Drug_input, Protein_input,Drug_Drug_input,Protein_Protein_input], outputs=[FC2])
model = Model(inputs=[Drug_input, Protein_input,Drug_Drug_input,Protein_Protein_input], outputs=[FC2, predictions])
full_model1 = Model(inputs=[Drug_input, Protein_input,Drug_Drug_input,Protein_Protein_input], outputs=[FC2, predictions])
full_model2 = Model(inputs=[Drug_input, Protein_input,Drug_Drug_input,Protein_Protein_input], outputs=[ predictions])
print(embedding_model.summary())
print(model.summary())

### define loss function and optimization algorithm for model

METRICS = [
      #keras.metrics.TruePositives(name='tp'),
      #keras.metrics.FalsePositives(name='fp'),
      #keras.metrics.TrueNegatives(name='tn'),
      #keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR') # precision-recall curve
]
#adam=Adam(lr=0.001)
def custom_contrastive_loss(y_true, y_pred):
  return multiclass_npairs_loss(y_pred, y_true)  # change this line based on the desired contrastive loss function
model.compile(optimizer='adam', loss=[custom_contrastive_loss, 'binary_crossentropy'], metrics={'dense4':[METRICS]})
es = EarlyStopping(monitor='prc', mode='max', verbose=1, patience=15)


##############################################################
#       Train and Validation Step                            #
##############################################################

positive_samples_d=np.array(positive_samples_d)
positive_samples_p=np.array(positive_samples_p)

negative_samples_d=np.array(negative_samples_d)
negative_samples_p=np.array(negative_samples_p)

smiles_rep=np.array(smiles_rep)
protein_rep=np.array(protein_rep)

margin_=0.2

kf = KFold(n_splits=10)
kf.get_n_splits(smiles_rep)

count1=0
for train_index_p, test_index_p in kf.split(positive_samples_d_orig):
    count2=0
    for train_index_n, test_index_n in kf.split(negative_samples_d_orig):
      if count1==count2:
         positive_samples_d_orig_train, positive_samples_d_orig_test = positive_samples_d_orig[train_index_p], positive_samples_d_orig[test_index_p]
         positive_samples_p_orig_train, positive_samples_p_orig_test = positive_samples_p_orig[train_index_p], positive_samples_p_orig[test_index_p]
         negative_samples_d_orig_train, negative_samples_d_orig_test = negative_samples_d_orig[train_index_n], negative_samples_d_orig[test_index_n]
         negative_samples_p_orig_train, negative_samples_p_orig_test = negative_samples_p_orig[train_index_n], negative_samples_p_orig[test_index_n]

         samples_d_orig_train = np.concatenate([positive_samples_d_orig_train, negative_samples_d_orig_train])
         samples_p_orig_train = np.concatenate([positive_samples_p_orig_train, negative_samples_p_orig_train])
         samples_d_orig_test = np.concatenate([positive_samples_d_orig_test, negative_samples_d_orig_test])
         samples_p_orig_test = np.concatenate([positive_samples_p_orig_test, negative_samples_p_orig_test])

         lbl = np.zeros((len(samples_d_orig_train),))
         lbl[0:len(positive_samples_d_orig_train)] = 1
         lbl_test = np.zeros((len(samples_d_orig_test),))
         lbl_test[0:len(positive_samples_d_orig_test)] = 1

         model.fit([np.squeeze(smiles_rep[samples_d_orig_train]),
                          np.squeeze(protein_rep[samples_p_orig_train]),
                          np.squeeze(drug_embeddings[samples_d_orig_train]),
                          np.squeeze(protein_embeddings[samples_p_orig_train])], [lbl, lbl],
                              epochs=5,
                               validation_data=[[np.squeeze(smiles_rep[samples_d_orig_test]),
                          np.squeeze(protein_rep[samples_p_orig_test]),
                          np.squeeze(drug_embeddings[samples_d_orig_test]),
                          np.squeeze(protein_embeddings[samples_p_orig_test])], [lbl_test, lbl_test]])



