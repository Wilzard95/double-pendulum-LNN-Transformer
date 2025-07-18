import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
import gc
import math # Para nn.TransformerEncoderLayer

# --- 1. Configuración ---
DATA_FOLDER = r"C:\Users\Nitro\Documents\Documentos William\Creando_mi_IA\Probando_modelos\Pendulos\dataset_pendulo_doble_500"

# --- !!! NUEVOS PARÁMETROS PARA TRANSFORMER !!! ---
SEQUENCE_LENGTH = 5       # Número de pasos de tiempo en cada secuencia de entrada
D_MODEL = 128             # Dimensión del modelo del Transformer (debe ser divisible por NHEAD)
NHEAD = 4                 # Número de cabezas en la atención multi-cabeza
NUM_ENCODER_LAYERS = 2    # Número de capas en el Transformer Encoder
DIM_FEEDFORWARD = 256     # Dimensión de la capa feedforward interna del Transformer
DROPOUT_TRANSFORMER = 0.1 # Dropout para el Transformer
# --- FIN NUEVOS PARÁMETROS ---

HIDDEN_DIM_CLASSIFIER = 128 # Para la cabeza MLP después del Transformer (si se usa)
MODEL_SAVE_PATH = os.path.join(DATA_FOLDER, f"lnn_transformer_seq{SEQUENCE_LENGTH}_d{D_MODEL}_h{NHEAD}_l{NUM_ENCODER_LAYERS}.pth")

NUM_EPOCHS = 35 # Reducir para pruebas iniciales con Transformer
BATCH_SIZE = 64  # Reducir batch size si hay problemas de memoria con secuencias
LEARNING_RATE = 1e-4

TRAIN_FILE_INDICES = range(0, 400)
VAL_FILE_INDICES = range(400, 450)
TEST_FILE_INDICES = range(450, 500)

FEATURE_COLUMNS = ['theta1_rad', 'omega1_rad_s', 'theta2_rad', 'omega2_rad_s']
KINETIC_ENERGY_COL = 'T_J'
POTENTIAL_ENERGY_COL = 'V_J'
INPUT_DIM_PER_STEP = len(FEATURE_COLUMNS) # Dimensión de cada estado (q, q_dot) = 4

# --- 2. Configuración de Dispositivo ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🚀 Usando GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️ GPU no encontrada, usando CPU.")

# --- 3. Función para obtener rutas de archivo (Sin cambios) ---
def get_filepaths(indices, folder):
    filepaths = [os.path.join(folder, f"sim_data_{i:03d}.txt") for i in indices]
    return filepaths

# --- 4. Clase Dataset (MODIFICADA para generar SECUENCIAS) ---
class LagrangianSequenceDataset(Dataset):
    def __init__(self, file_paths, sequence_length):
        self.file_paths = file_paths
        self.sequence_length = sequence_length
        if not self.file_paths:
            raise ValueError("La lista de archivos proporcionada está vacía.")
        print(f"  Cargando {len(self.file_paths)} archivo(s) para secuencias de longitud {sequence_length}...")
        
        all_sequences = []
        all_targets = [] # Lagrangiano del ÚLTIMO elemento de la secuencia
        loaded_count = 0
        columns_to_read = FEATURE_COLUMNS + [KINETIC_ENERGY_COL, POTENTIAL_ENERGY_COL]

        for fpath in tqdm(self.file_paths, desc="  Leyendo y creando secuencias", leave=False):
            if not os.path.exists(fpath):
                print(f"\nAdvertencia: Archivo no encontrado, saltando: {fpath}")
                continue
            try:
                df = pd.read_csv(fpath, sep='\t', comment='#', header=0, encoding='utf-8')
                if df.empty or len(df) < self.sequence_length:
                    print(f"\nAdvertencia: Archivo {fpath} vacío o demasiado corto ({len(df)} puntos) para secuencias de longitud {self.sequence_length}. Saltando.")
                    continue
                
                if not all(col in df.columns for col in columns_to_read):
                    missing = [col for col in columns_to_read if col not in df.columns]
                    print(f"\nAdvertencia: Faltan columnas {missing} en {fpath}. Saltando.")
                    continue

                features_np = df[FEATURE_COLUMNS].values.astype(np.float32)
                kinetic_energy = df[KINETIC_ENERGY_COL].values.astype(np.float32)
                potential_energy = df[POTENTIAL_ENERGY_COL].values.astype(np.float32)
                lagrangian_np = kinetic_energy - potential_energy

                # Crear secuencias deslizantes
                for i in range(len(features_np) - self.sequence_length + 1):
                    seq = features_np[i : i + self.sequence_length]
                    target_lagrangian = lagrangian_np[i + self.sequence_length - 1] # L del último elemento
                    
                    all_sequences.append(seq)
                    all_targets.append(target_lagrangian)
                loaded_count += 1

            except Exception as e:
                print(f"\nError procesando {fpath}: {e}")
                import traceback
                traceback.print_exc()

        if not all_sequences:
            raise ValueError("No se pudieron crear secuencias válidas de los archivos especificados.")

        print(f"\n  Se procesaron datos de {loaded_count}/{len(self.file_paths)} archivos.")
        print("  Convirtiendo a tensores...")
        
        self.sequences = torch.tensor(np.array(all_sequences, dtype=np.float32), dtype=torch.float32)
        self.targets = torch.tensor(np.array(all_targets, dtype=np.float32), dtype=torch.float32).unsqueeze(1) # (N, 1)
        
        del all_sequences, all_targets # Liberar memoria de listas de Python
        gc.collect()
        
        print(f"  Total de secuencias en este dataset: {len(self.sequences):,}")
        print("  Forma de entrada (secuencias):", self.sequences.shape) # Debería ser (N_seq, sequence_length, num_features)
        print("  Forma de salida (targets):", self.targets.shape)     # Debería ser (N_seq, 1)
        print("  Dataset listo.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# --- 5. Definición de la Red Neuronal Lagrangiana con Transformer ---

class PositionalEncoding(nn.Module):
    # Tomado de los tutoriales de PyTorch: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape (max_len, 1, d_model) -> (seq_len, batch, d_model)
        self.register_buffer('pe', pe) # No es un parámetro del modelo

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LagrangianTransformer(nn.Module):
    def __init__(self, input_dim_per_step, d_model, nhead, num_encoder_layers, dim_feedforward, 
                 sequence_length, dropout=0.1, hidden_dim_classifier=128):
        super(LagrangianTransformer, self).__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length

        # 1. Capa de entrada: proyecta cada paso de la secuencia a d_model
        self.input_projection = nn.Linear(input_dim_per_step, d_model)
        
        # 2. Codificación Posicional
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=sequence_length + 1) # +1 por si acaso

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, dropout=dropout, 
                                                   batch_first=False) # Espera (seq_len, batch, features)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 4. Cabeza de Clasificación/Regresión (MLP) para predecir el Lagrangiano
        # Toma la salida del Transformer para el ÚLTIMO paso de tiempo
        self.output_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim_classifier), # d_model (del último paso) -> hidden
            nn.Softplus(), # o ReLU
            nn.Linear(hidden_dim_classifier, 1)  # hidden -> 1 (Lagrangiano)
        )
        
        print(f"--- Creando LNN con Transformer Encoder ---")
        print(f"  Input dim/step: {input_dim_per_step}, Proyección a d_model: {d_model}")
        print(f"  Sequence Length: {sequence_length}")
        print(f"  Transformer: {num_encoder_layers} layers, {nhead} heads, dim_ff: {dim_feedforward}")
        print(f"  Output MLP hidden: {hidden_dim_classifier}")
        print(f"-------------------------------------------")

    def forward(self, src_sequence):
        # src_sequence shape: (batch_size, seq_len, input_dim_per_step)
        
        # Proyectar cada elemento de la secuencia a d_model
        # (batch_size, seq_len, input_dim_per_step) -> (batch_size, seq_len, d_model)
        projected_src = self.input_projection(src_sequence) 
        
        # El TransformerEncoder de PyTorch espera (seq_len, batch_size, d_model) por defecto
        # o (batch_size, seq_len, d_model) si batch_first=True en TransformerEncoderLayer y TransformerEncoder
        # Aquí estamos usando batch_first=False (por defecto) para TransformerEncoderLayer
        projected_src = projected_src.transpose(0, 1) # (seq_len, batch_size, d_model)
        
        # Añadir codificación posicional
        # projected_src shape: (seq_len, batch_size, d_model)
        src_with_pos = self.pos_encoder(projected_src * math.sqrt(self.d_model)) # Escalar antes de pos_enc, común en Transformers

        # Pasar por el Transformer Encoder
        # src_with_pos shape: (seq_len, batch_size, d_model)
        # memory shape: (seq_len, batch_size, d_model)
        memory = self.transformer_encoder(src_with_pos)
        
        # Queremos el Lagrangiano correspondiente al ÚLTIMO estado de la secuencia
        # Tomamos la salida del Transformer para el último token de la secuencia
        # memory shape: (seq_len, batch_size, d_model)
        last_step_output = memory[-1, :, :] # Shape: (batch_size, d_model)
        
        # Pasar por la MLP de salida
        lagrangian_pred = self.output_mlp(last_step_output) # Shape: (batch_size, 1)
        
        return lagrangian_pred

# --- 6. Entrenamiento y Evaluación (adaptaciones menores) ---
if __name__ == '__main__':
    start_time = time.time()

    print("\n--- Preparando Listas de Archivos para los Conjuntos ---")
    train_files = get_filepaths(TRAIN_FILE_INDICES, DATA_FOLDER)
    val_files = get_filepaths(VAL_FILE_INDICES, DATA_FOLDER)
    test_files = get_filepaths(TEST_FILE_INDICES, DATA_FOLDER)
    # ... (mensajes de impresión sin cambios)

    print("\n--- Creando Datasets y DataLoaders ---")
    try:
        print(" Cargando datos de Entrenamiento...")
        train_data = LagrangianSequenceDataset(file_paths=train_files, sequence_length=SEQUENCE_LENGTH)
        print("\n Cargando datos de Validación...")
        val_data = LagrangianSequenceDataset(file_paths=val_files, sequence_length=SEQUENCE_LENGTH)
    except ValueError as e:
        print(f"\nError fatal al crear datasets: {e}"); exit()
    except MemoryError:
        print("\n--- ERROR DE MEMORIA AL CARGAR DATASETS ---"); exit()

    num_workers = 0 
    pin_memory_flag = torch.cuda.is_available()
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_memory_flag, persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory_flag, persistent_workers=(num_workers > 0))
    # ... (mensajes de impresión sin cambios)

    print("\n--- Configurando Modelo ---")
    model = LagrangianTransformer(input_dim_per_step=INPUT_DIM_PER_STEP, 
                                  d_model=D_MODEL, nhead=NHEAD, 
                                  num_encoder_layers=NUM_ENCODER_LAYERS, 
                                  dim_feedforward=DIM_FEEDFORWARD,
                                  sequence_length=SEQUENCE_LENGTH,
                                  dropout=DROPOUT_TRANSFORMER,
                                  hidden_dim_classifier=HIDDEN_DIM_CLASSIFIER
                                  ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True) # Paciencia un poco mayor para Transformer
    # ... (mensajes de parámetros y scheduler sin cambios)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Número total de parámetros entrenables: {total_params:,}")


    print("\n--- Iniciando Entrenamiento ---")
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        model.train()
        running_train_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)
        for sequences, targets in train_progress_bar: # Ahora son secuencias
            sequences = sequences.to(device, non_blocking=pin_memory_flag and num_workers > 0)
            targets = targets.to(device, non_blocking=pin_memory_flag and num_workers > 0)
            
            optimizer.zero_grad()
            outputs = model(sequences) # El modelo ahora toma secuencias
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clipping de gradiente, puede ayudar con Transformers
            optimizer.step()
            
            running_train_loss += loss.item()
            train_progress_bar.set_postfix(loss=f"{loss.item():.6f}")
        avg_train_loss = running_train_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", leave=False)
        with torch.no_grad():
            for sequences, targets in val_progress_bar:
                sequences = sequences.to(device, non_blocking=pin_memory_flag and num_workers > 0)
                targets = targets.to(device, non_blocking=pin_memory_flag and num_workers > 0)
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()
                val_progress_bar.set_postfix(loss=f"{loss.item():.6f}")
        avg_val_loss = running_val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start_time

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - "
              f"Train Loss: {avg_train_loss:.6f} - "
              f"Val Loss: {avg_val_loss:.6f} - "
              f"LR: {current_lr:.1e} - "
              f"Tiempo: {epoch_time:.2f}s")

        if avg_val_loss < best_val_loss:
            print(f"   ✨ Mejor Val Loss encontrado: {avg_val_loss:.6f} (anterior: {best_val_loss:.6f}). Guardando modelo...")
            best_val_loss = avg_val_loss
            try:
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
            except Exception as e:
                print(f"Error al guardar el modelo: {e}")
        
        if device.type == 'cuda': torch.cuda.empty_cache()
        gc.collect()

    # ... (Fin del entrenamiento e impresión de resultados sin cambios) ...

    # --- EVALUACIÓN EN TEST SET (adaptar para cargar el modelo Transformer) ---
    print("\n--- Evaluando en el Conjunto de Prueba ---")
    try:
        print(" Cargando el mejor modelo Transformer guardado...")
        model_test = LagrangianTransformer(input_dim_per_step=INPUT_DIM_PER_STEP, 
                                           d_model=D_MODEL, nhead=NHEAD, 
                                           num_encoder_layers=NUM_ENCODER_LAYERS, 
                                           dim_feedforward=DIM_FEEDFORWARD,
                                           sequence_length=SEQUENCE_LENGTH,
                                           dropout=DROPOUT_TRANSFORMER,
                                           hidden_dim_classifier=HIDDEN_DIM_CLASSIFIER
                                           ).to(device)
        model_test.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        model_test.eval()

        print(" Cargando datos de Prueba (secuencias)...")
        try:
            test_data = LagrangianSequenceDataset(file_paths=test_files, sequence_length=SEQUENCE_LENGTH)
            test_loader = DataLoader(test_data, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory_flag, persistent_workers=(num_workers > 0))
            # ... (mensajes de DataLoader sin cambios)

            test_loss = 0.0
            test_progress_bar = tqdm(test_loader, desc="[Test]", leave=True)
            with torch.no_grad():
                for sequences, targets in test_progress_bar:
                    sequences = sequences.to(device, non_blocking=pin_memory_flag and num_workers > 0)
                    targets = targets.to(device, non_blocking=pin_memory_flag and num_workers > 0)
                    outputs = model_test(sequences)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()
                    test_progress_bar.set_postfix(loss=f"{loss.item():.6f}")

            avg_test_loss = test_loss / len(test_loader)
            print(f"\n📈 Pérdida MSE en el Conjunto de Prueba (Transformer): {avg_test_loss:.6f}")

        except ValueError as e:
             print(f"\nError al cargar datos de prueba: {e}")
        # ... (resto del bloque de Test y finally sin cambios significativos, solo asegurar limpieza) ...
    except FileNotFoundError:
         print(f"Error: No se encontró el archivo del modelo guardado en {MODEL_SAVE_PATH}.")
    except Exception as e:
         print(f"\nError inesperado durante la evaluación del Test Set: {e}")
         import traceback; traceback.print_exc()
    finally:
        if 'model' in locals(): del model
        if 'model_test' in locals(): del model_test
        if 'train_data' in locals(): del train_data
        if 'val_data' in locals(): del val_data
        if 'test_data' in locals(): del test_data
        if 'train_loader' in locals(): del train_loader
        if 'val_loader' in locals(): del val_loader
        if 'test_loader' in locals(): del test_loader
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()