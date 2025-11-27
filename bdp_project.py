# -*- coding: utf-8 -*-
"""BDP-Project Docker-ready version - Logging & flush improved"""

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pyspark.sql import SparkSession
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, NGram, HashingTF, IDF
from pyspark.ml import Pipeline
from pyspark.sql.functions import concat, col
import tensorflow as tf
import time
import joblib
from sklearn.metrics import accuracy_score, recall_score

tqdm.pandas()

# --------------------------
# Função de log customizada
# --------------------------
def log(msg):
    print(f"\033[94m[INFO]\033[0m {msg}", flush=True)

# --------------------------
# 1️⃣ Definir paths
# --------------------------
DATA_DIR = '/opt/spark-apps/Data'
OUTPUT_DIR = '/opt/spark-apps/Results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

log(f"DATA_DIR = {DATA_DIR}")
log(f"OUTPUT_DIR = {OUTPUT_DIR}")

# --------------------------
# 2️⃣ Função para limpeza de texto
# --------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

# --------------------------
# 3️⃣ Carregar datasets
# --------------------------
fake_path = os.path.join(DATA_DIR,'fake-and-real-news-dataset', 'Fake.csv')
true_path = os.path.join(DATA_DIR,'fake-and-real-news-dataset', 'True.csv')
liar_train = os.path.join(DATA_DIR, 'LIAR-DATASET', 'train.tsv')
liar_test  = os.path.join(DATA_DIR, 'LIAR-DATASET', 'test.tsv')
liar_valid = os.path.join(DATA_DIR, 'LIAR-DATASET', 'valid.tsv')

log("Verificando paths dos datasets:")
for p in [fake_path, true_path, liar_train, liar_test, liar_valid]:
    log(f"{p} exists? {os.path.exists(p)}")

# Load Fake/True
fake_df = pd.read_csv(fake_path)
true_df = pd.read_csv(true_path)
fake_df['label'] = 0
true_df['label'] = 1

# Load LIAR
col_names = ["id","label","statement","subject","speaker","speaker_job_title","state_info","party_affiliation",
             "barely_true_counts","false_counts","half_true_counts","mostly_true_counts","pants_on_fire_counts","context"]
liar_parts = []
for p in [liar_train, liar_test, liar_valid]:
    if os.path.exists(p):
        tmp = pd.read_csv(p, sep='\t', header=None, quoting=3, encoding='latin1', engine='python')
        tmp.columns = col_names[:tmp.shape[1]]
        liar_parts.append(tmp)
liar_df = pd.concat(liar_parts, ignore_index=True) if liar_parts else pd.DataFrame(columns=col_names)

# Map LIAR labels to binary
if not liar_df.empty:
    liar_df['label'] = liar_df['label'].apply(lambda x: 1 if str(x).lower()=='true' else 0)

# Prepare 'text' column
if 'text' not in fake_df.columns and 'title' in fake_df.columns:
    fake_df['text'] = fake_df['title'] + ' ' + fake_df.get('text','')
if 'text' not in true_df.columns and 'title' in true_df.columns:
    true_df['text'] = true_df['title'] + ' ' + true_df.get('text','')

fake_small = fake_df[['text','label']].copy()
true_small = true_df[['text','label']].copy()
liar_small = liar_df[['statement','label']].rename(columns={'statement':'text'}).copy() if not liar_df.empty else pd.DataFrame(columns=['text','label'])

all_df = pd.concat([fake_small, true_small, liar_small], ignore_index=True)
all_df['clean_text'] = all_df['text'].progress_apply(clean_text)
all_df = all_df[all_df['clean_text'].str.len() > 0].reset_index(drop=True)

# Save cleaned dataset
cleaned_path = os.path.join(OUTPUT_DIR, 'combined_cleaned.csv')
try:
    all_df.to_csv(cleaned_path, index=False)
    log(f"Saved cleaned dataset to {cleaned_path}")
except Exception as e:
    log(f"Erro ao salvar cleaned dataset: {e}")

# --------------------------
# 4️⃣ Spark Session
# --------------------------
spark = SparkSession.builder \
    .appName('fake-news-preprocessing-ngram') \
    .master('spark://spark-master:7077') \
    .config('spark.executor.memory', '6g') \
    .config('spark.driver.memory', '6g') \
    .getOrCreate()
log(f"Spark version: {spark.version}")

# Convert Pandas → Spark
sdf = spark.createDataFrame(all_df[['clean_text', 'label']].rename(columns={'clean_text': 'text'}))
log(f"Input size: {sdf.count():,} rows")

# --------------------------
# 5️⃣ Spark preprocessing pipeline
# --------------------------
tokenizer = RegexTokenizer(inputCol='text', outputCol='tokens', pattern='\\W')
remover = StopWordsRemover(inputCol='tokens', outputCol='filtered')
ngram2 = NGram(n=2, inputCol='filtered', outputCol='bigrams')

def merge_columns(df):
    return df.withColumn("tokens_final", concat(col("filtered"), col("bigrams")))

hashTF = HashingTF(inputCol='tokens_final', outputCol='rawFeatures', numFeatures=8000)
idf = IDF(inputCol='rawFeatures', outputCol='tfidf_features')

pipeline_initial = Pipeline(stages=[tokenizer, remover, ngram2])
model_initial = pipeline_initial.fit(sdf)
sdf_ng = model_initial.transform(sdf)
sdf_ng = merge_columns(sdf_ng)

pipeline_tfidf = Pipeline(stages=[hashTF, idf])
model_tfidf = pipeline_tfidf.fit(sdf_ng)
sdf_feat = model_tfidf.transform(sdf_ng)

#sdf_sample = sdf_feat.sample(fraction=0.50, seed=42)
pdf = sdf_feat.select("tfidf_features", "label").toPandas()
log(f"TF-IDF size = {len(pdf)}")

X_tfidf = np.vstack(pdf['tfidf_features'].apply(lambda v: v.toArray()).values)
y = pdf['label'].values

try:
    np.save(os.path.join(OUTPUT_DIR, 'X_tfidf.npy'), X_tfidf)
    np.save(os.path.join(OUTPUT_DIR, 'y_labels.npy'), y)
    log("Saved TF-IDF features.")
except Exception as e:
    log(f"Erro ao salvar TF-IDF: {e}")

# --------------------------
# 6️⃣ LSTM tokenizer and sequences
# --------------------------
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

NUM_WORDS = 10000
MAX_LEN = 200

tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(tqdm(all_df['clean_text'], desc="Tokenizer fitting"))
sequences = tokenizer.texts_to_sequences(tqdm(all_df['clean_text'], desc="Text to sequences"))
X_seq = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
y_seq = all_df['label'].values

try:
    np.save(os.path.join(OUTPUT_DIR,'X_seq.npy'), X_seq)
    np.save(os.path.join(OUTPUT_DIR,'y_seq.npy'), y_seq)
    with open(os.path.join(OUTPUT_DIR,'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)
    log("Saved sequences and tokenizer.")
except Exception as e:
    log(f"Erro ao salvar sequences/tokenizer: {e}")

# --------------------------
# 7️⃣ Parallel LSTM training with Spark RDD
# --------------------------
sc = spark.sparkContext

configs = [
    {'name': 'lstm_64',  'units': 64,  'drop': 0.3, 'epochs': 2, 'batch_size': 128},
    {'name': 'lstm_128', 'units': 128, 'drop': 0.4, 'epochs': 2, 'batch_size': 128},
    {'name': 'lstm_256', 'units': 256, 'drop': 0.5, 'epochs': 1, 'batch_size': 128}
]

with open(os.path.join(OUTPUT_DIR, 'configs.json'), 'w') as f:
    json.dump(configs, f)

log("Iniciando treinamento paralelo em Spark RDD...")

def train_lstm_worker(cfg):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import os

    try:
        X = np.load('/opt/spark-apps/Results/X_seq.npy', mmap_mode='r')
        y = np.load('/opt/spark-apps/Results/y_seq.npy', mmap_mode='r')
        # Reduz dataset se for muito grande
        if len(X) > 50000:
            idx = np.random.choice(len(X), size=40000, replace=False)
            X, y = X[idx], y[idx]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        model = Sequential([
            Embedding(input_dim=10000, output_dim=128, input_length=X.shape[1]),
            LSTM(cfg['units']),
            Dropout(cfg['drop']),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=cfg['epochs'], batch_size=cfg['batch_size'], verbose=1)

        preds = (model.predict(X_val, batch_size=512, verbose=0) > 0.5).astype(int).flatten()
        acc = float(accuracy_score(y_val, preds))
        weights_path = os.path.join('/opt/spark-apps/Results', cfg['name'] + '.weights.h5')
        model.save_weights(weights_path)
        tf.keras.backend.clear_session()
        log(f"Finished {cfg['name']} with acc={acc:.4f}")
        return {'name': cfg['name'], 'acc': acc, 'weights': weights_path}
    except Exception as e:
        log(f"Error in {cfg['name']}: {e}")
        return {'name': cfg['name'], 'error': str(e)}

try:
    # Cria RDD com partições suficientes
    rdd = sc.parallelize(configs, len(configs))
    results = rdd.map(train_lstm_worker).collect()  # collect após map garante que tudo termine
    with open(os.path.join(OUTPUT_DIR, 'parallel_results.json'), 'w') as f:
        json.dump(results, f)
    log("✅ Parallel training finished. Results saved.")
except Exception as e:
    log(f"Erro durante treinamento paralelo: {e}")



#--------------------------------------------
# Evaluating the models
#--------------------------------------


start_eval = time.time()

# Load data
X_tfidf = np.load(os.path.join(OUTPUT_DIR, 'X_tfidf.npy'), mmap_mode='r')
y_true = np.load(os.path.join(OUTPUT_DIR, 'y_labels.npy'), mmap_mode='r')
X_seq = np.load(os.path.join(OUTPUT_DIR, 'X_seq.npy'), mmap_mode='r')

idx = np.random.choice(len(X_tfidf), 5000, replace=False)
X_tfidf_test = X_tfidf[idx]
y_test = y_true[idx]
X_seq_test = X_seq[idx]

# Load models
models = {}

# LSTM loaders
def build_lstm(units):
    m = tf.keras.Sequential([
        tf.keras.layers.Embedding(10000, 128),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    m.compile(loss='binary_crossentropy', optimizer='adam')
    return m

lstm_paths = [
    ("lstm64", os.path.join(OUTPUT_DIR, "lstm_64.weights.h5"), 64),
    ("lstm128", os.path.join(OUTPUT_DIR, "lstm_128.weights.h5"), 128),
    ("lstm256", os.path.join(OUTPUT_DIR, "lstm_256.weights.h5"), 256),
]

lstm_models = {}
for name, path, units in lstm_paths:
    if os.path.exists(path):
        model = build_lstm(units)
        model.build(input_shape=(None, X_seq_test.shape[1])) # Build the model with input shape
        model.load_weights(path)
        lstm_models[name] = model
    else:
        print(f"LSTM model weights not found: {path}")

# Predict with timing
model_times = {}
preds = {}

for name, model in models.items():
    t0 = time.time()
    preds[name] = model.predict(X_tfidf_test)
    model_times[name] = time.time() - t0

for name, model in lstm_models.items():
    t0 = time.time()
    p = (model.predict(X_seq_test, batch_size=256, verbose=0) > 0.5).astype(int).flatten()
    preds[name] = p
    model_times[name] = time.time() - t0

# Load individual model training results
with open(os.path.join(OUTPUT_DIR, 'parallel_results.json'), 'r') as f:
    training_results = json.load(f)

# Create a dictionary for easier lookup of training accuracies
training_acc_map = {res['name']: res['acc'] for res in training_results}

print("\n📊 Individual Model Performance:")
for name_short, path, units in lstm_paths:
    model_name = name_short.replace('lstm','lstm_') # Adjust name to match stored results
    if model_name in training_acc_map and name_short in preds:
        train_acc = training_acc_map[model_name]
        eval_preds = preds[name_short]
        eval_acc = accuracy_score(y_test, eval_preds)
        proc_time = model_times.get(name_short, 0.0)
        print(f"{name_short.upper()} \n  Training Accuracy: {train_acc:.4f}\n  Evaluation Accuracy: {eval_acc:.4f}\n  Processing Time: {proc_time:.3f}s\n")
    else:
        print(f"Warning: Could not find full data for {model_name}")

# Hybrid majority voting
if len(preds) == 0:
    print("No models were loaded for evaluation. Cannot perform ensemble prediction.")
    ensemble_pred = np.array([])
elif len(preds) == 1:
    print("Only one model loaded for evaluation, using its predictions as ensemble.")
    ensemble_pred = list(preds.values())[0]
else:
    all_preds = np.array(list(preds.values()))
    ensemble_pred = (np.sum(all_preds, axis=0) >= (all_preds.shape[0] / 2)).astype(int)

end_eval = time.time()

# Metrics
if ensemble_pred.size > 0:
    acc = accuracy_score(y_test, ensemble_pred)
    rec = recall_score(y_test, ensemble_pred)

    print("\n📊 Ensemble Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Recall: {rec:.4f}")
else:
    print("\n📊 Ensemble Results: No predictions to evaluate.")

print(f"Total evaluation time: {end_eval - start_eval:.2f} seconds")
print("\n⏱ Model processing times:")
for name, t in model_times.items():
    print(f"{name} \u2192 {t:.3f}s")




# --------------------------
# ✅ Fim do script
# --------------------------
log("Script Docker-ready concluído!")
