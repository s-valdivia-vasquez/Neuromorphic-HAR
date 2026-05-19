from __future__ import annotations
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, List
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset

def load_csv(csv_path: Union[str, Path]) -> np.ndarray:
    """
    Carga CSV IMU y retorna señal contigua [T,6] int32
    con columnas fijas: ax, ay, az, gx, gy, gz.
    """
    csv_path = Path(csv_path)
    cols = ["ax", "ay", "az", "gx", "gy", "gz"]
    df = pd.read_csv(csv_path, usecols=cols)
    return df.to_numpy(dtype=np.int32, copy=False)

def load_batches(
    csv_path: str,
    batch_size: int = 512,
    overlap: int = 256,
    channels: list = ["ax", "ay", "az", "gx", "gy", "gz"],
) -> np.ndarray:
    """
    Carga un CSV IMU y lo segmenta en batches con solapamiento.

    Parámetros
    ----------
    csv_path : str
        Ruta al archivo CSV.
    batch_size : int
        Tamaño de ventana (por defecto 512).
    overlap : int
        Solapamiento entre ventanas (por defecto 256).
    channels : list
        Columnas a extraer en el orden deseado.

    Retorna
    -------
    np.ndarray
        Tensor (N, batch_size, 6) con N ventanas completas.
    """

    if overlap < 0 or overlap >= batch_size:
        raise ValueError("overlap debe cumplir: 0 <= overlap < batch_size")

    stride = batch_size - overlap

    df = pd.read_csv(csv_path)

    missing_cols = set(channels) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Columnas faltantes en el CSV: {missing_cols}")

    data = df[channels].to_numpy(dtype=np.int32)
    total_samples = data.shape[0]

    # Cantidad de ventanas completas: start+batch_size <= total_samples
    if total_samples < batch_size:
        return np.empty((0, batch_size, len(channels)), dtype=np.int32)

    num_batches = 1 + (total_samples - batch_size) // stride

    batches = np.empty((num_batches, batch_size, len(channels)), dtype=np.int32)
    for i in range(num_batches):
        start = i * stride
        end = start + batch_size
        batches[i] = data[start:end]

    return batches

def csv_paths_and_labels(
    dataset_root: Union[str, Path],
    body_zone: str = "tobillo_derecho",
    class_map: Optional[Dict[int, str]] = None,
    captures_glob: str = "imu_capturas*",
) -> Tuple[List[str], List[int], Dict[int, str]]:
    """
    Enlista rutas a CSVs de una zona (por prefijo) y retorna labels numéricas + class_map.

    Por defecto, class_map es:
        id_clase -> nombre_clase

    Si class_map es None:
      - Detecta clases como carpetas directas en dataset_root (solo nivel 1).
      - Asigna IDs ordenando alfabéticamente los nombres de carpeta
        para asegurar reproducibilidad.

    Estructura esperada:
      dataset_root/
        <ClaseA>/
          imu_capturas1/
            tobillo_derecho1/tobillo_derecho1.csv
        <ClaseB>/
          imu_capturasX/
            tobillo_derecho2/tobillo_derecho2.csv
        ...

    Parámetros
    ----------
    dataset_root : str | Path
        Ruta base del dataset.
    body_zone : str
        Prefijo de la zona del cuerpo (ej. "tobillo_derecho").
    class_map : dict[int, str] | None
        Mapeo id_clase -> nombre_clase.
        Si None, se infiere automáticamente.
    captures_glob : str
        Patrón para carpetas de capturas (por defecto "imu_capturas*").

    Retorna
    -------
    paths : list[str]
        Rutas a los CSV encontrados (ordenadas).
    y : list[int]
        Etiquetas numéricas alineadas con paths.
    class_map : dict[int, str]
        Mapeo id_clase -> nombre_clase.
    """
    root = Path(dataset_root)

    if not root.exists() or not root.is_dir():
        raise NotADirectoryError(f"dataset_root no es un directorio válido: {root}")

    # 1) Inferir clases automáticamente si no vienen dadas
    if class_map is None:
        class_names = sorted(
            d.name for d in root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        class_map = {i: name for i, name in enumerate(class_names)}
    else:
        # Normaliza tipos
        class_map = {int(k): str(v) for k, v in class_map.items()}

    # Mapeo interno para recorrido: nombre_clase -> id_clase
    class_name_to_id = {v: k for k, v in class_map.items()}

    paths: List[str] = []
    y: List[int] = []

    # 2) Recorrer clases en orden estable por ID
    for class_id in sorted(class_map.keys()):
        class_name = class_map[class_id]
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue

        # 3) Entra a carpetas imu_capturas*
        for capture_dir in sorted(p for p in class_dir.glob(captures_glob) if p.is_dir()):
            # 4) Dentro de imu_capturasK, busca carpetas que comiencen con el prefijo de zona
            zone_dirs = sorted(
                d for d in capture_dir.iterdir()
                if d.is_dir() and d.name.startswith(body_zone)
            )

            for zone_dir in zone_dirs:
                csv_path = zone_dir / f"{zone_dir.name}.csv"
                if csv_path.is_file():
                    paths.append(str(csv_path))
                    y.append(class_id)

    return paths, y, class_map



def listar_csv_con_etiquetas(dataset_root: str) -> Tuple[List[str], List[int], Dict[str, int]]:
    """
    Recorre un dataset con estructura:
        dataset_root/
            clase_a/
                *.csv
            clase_b/
                *.csv
            ...

    Retorna:
        - rutas_csv:  lista de rutas absolutas a archivos .csv
        - etiquetas:  lista de etiquetas (int) alineadas 1:1 con rutas_csv, en [0, num_clases-1]
        - label_map:  dict {nombre_clase: etiqueta_int}

    Notas:
        - Soporta agregar nuevas clases en el futuro (subcarpetas nuevas).
        - Ignora archivos no .csv y subcarpetas internas dentro de cada clase.
        - El mapeo de etiquetas se define por orden alfabético de nombres de carpetas (reproducible).
    """
    if not isinstance(dataset_root, str) or not dataset_root.strip():
        raise ValueError("dataset_root debe ser un string no vacío.")

    root_abs = os.path.abspath(os.path.expanduser(dataset_root))
    if not os.path.isdir(root_abs):
        raise FileNotFoundError(f"No existe el directorio raíz del dataset: {root_abs}")

    # 1) Detectar clases como subcarpetas directas del root
    clases = []
    for name in os.listdir(root_abs):
        class_dir = os.path.join(root_abs, name)
        if os.path.isdir(class_dir):
            clases.append(name)

    if not clases:
        raise ValueError(f"No se encontraron carpetas de clases dentro de: {root_abs}")

    # Orden estable para que el label_map sea reproducible
    clases.sort()

    # 2) Construir label_map: {nombre_clase: idx}
    label_map: Dict[str, int] = {cls_name: idx for idx, cls_name in enumerate(clases)}

    # 3) Recolectar CSVs por clase
    rutas_csv: List[str] = []
    etiquetas: List[int] = []

    for cls_name in clases:
        cls_dir = os.path.join(root_abs, cls_name)
        y = label_map[cls_name]

        # Recorre solo archivos en el primer nivel de la carpeta de clase
        # (si necesitas recursivo dentro de subcarpetas, se puede cambiar a os.walk).
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(".csv"):
                rutas_csv.append(fpath)
                etiquetas.append(y)

    if not rutas_csv:
        raise ValueError(f"No se encontraron archivos .csv dentro de las carpetas de clases en: {root_abs}")

    return rutas_csv, etiquetas, label_map




def get_data(dataset_root: Union[str, Path],body_zone: str = "tobillo_derecho",batch_size: int = 64,splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),overlap: int = 256):
    
    paths, y, class_map = csv_paths_and_labels(dataset_root=dataset_root, body_zone=body_zone)

    X_list = []
    Y_list = []

    for csv_path, label in zip(paths, y):
        b = load_batches(csv_path,batch_size=512,overlap=overlap)  # (N, 512, 6)
        if b is None or b.shape[0] == 0:
            continue
        X_list.append(b.astype(np.int32, copy=False))
        Y_list.append(np.full((b.shape[0],), label, dtype=np.int64))

    if not X_list:
        raise RuntimeError("No se cargaron batches válidos desde el dataset.")

    X = np.concatenate(X_list, axis=0)  # (N, T, C)
    Y = np.concatenate(Y_list, axis=0)  # (N,)

    tr, va, te = splits
    tr = float(tr); va = float(va); te = float(te)
    if tr < 0 or va < 0 or te < 0:
        raise ValueError("splits no puede tener valores negativos.")
    s = tr + va + te
    if s <= 0:
        raise ValueError("La suma de splits debe ser > 0.")
    tr, va, te = tr / s, va / s, te / s

    rng = np.random.default_rng(42)

    train_idx = []
    val_idx = []
    test_idx = []

    for class_id in np.unique(Y):
        cls_idx = np.where(Y == class_id)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)

        if n == 0:
            continue

        # asignación aproximada por clase
        n_val = int(round(n * va))
        n_test = int(round(n * te))

        # Asegurar "al menos 1" en val/test si se pidió y si es factible
        # (no vamos a vaciar train en clases minúsculas)
        if va > 0 and n >= 2:
            n_val = max(1, n_val)
        if te > 0 and n >= 3:
            n_test = max(1, n_test)

        # No exceder n-1 para que quede algo en train, cuando sea posible
        if n >= 2:
            n_val = min(n_val, n - 1)
        else:
            n_val = 0  # n=1 -> no se puede poner en val sin dejar train vacío

        if n >= 3:
            n_test = min(n_test, n - 1 - n_val)  # deja al menos 1 en train
        else:
            n_test = 0  # n=1 o 2 -> test es difícil sin dejar train muy chico

        # cortes
        v_end = n_val
        t_end = n_val + n_test

        val_idx.append(cls_idx[:v_end])
        test_idx.append(cls_idx[v_end:t_end])
        train_idx.append(cls_idx[t_end:])

    train_idx = np.concatenate(train_idx) if train_idx else np.array([], dtype=np.int64)
    val_idx   = np.concatenate(val_idx)   if val_idx   else np.array([], dtype=np.int64)
    test_idx  = np.concatenate(test_idx)  if test_idx  else np.array([], dtype=np.int64)

    # Shuffle interno de índices para no dejar agrupado por clase dentro del split
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val,   Y_val   = X[val_idx],   Y[val_idx]
    X_test,  Y_test  = X[test_idx],  Y[test_idx]

    class _HARDataset(Dataset):
        def __init__(self, X_np, Y_np):
            self.samples = X_np
            self.labels = Y_np

        def __len__(self):
            return int(self.labels.shape[0])

        def __getitem__(self, idx):
            sample = torch.from_numpy(self.samples[idx])  # [T, C]
            target = torch.tensor(int(self.labels[idx]), dtype=torch.long)
            return sample, target

    train_ds = _HARDataset(X_train, Y_train)
    val_ds   = _HARDataset(X_val, Y_val)
    test_ds  = _HARDataset(X_test, Y_test)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(train_ds,batch_size=batch_size,shuffle=True,drop_last=False,num_workers=0,pin_memory=pin)
    val_loader = DataLoader(val_ds,batch_size=batch_size,shuffle=False,drop_last=False,num_workers=0,pin_memory=pin)
    test_loader = DataLoader(test_ds,batch_size=batch_size,shuffle=False,drop_last=False,num_workers=0,pin_memory=pin)

    return train_loader, val_loader, test_loader, class_map

def make_dataloaders(
    X,
    y,
    split=(0.8, 0.1, 0.1),
    batch_size=64,
    seed=42,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
    shuffle_train=True,
):
    """
    Crea DataLoaders train/val/test con split estratificado (aprox. misma proporción por clase),
    aleatorio y reproducible.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        Datos con shape (N, ...). Ej: (1257, 512, 6, 2).
    y : np.ndarray or torch.Tensor
        Etiquetas con shape (N,). dtype int.
    split : tuple
        Proporciones (train, val, test). Debe sumar 1.0.
    batch_size : int
        Batch size para los 3 loaders (puedes cambiarlo luego si quieres).
    seed : int
        Semilla para reproducibilidad (split + shuffle de train).
    num_workers : int
        Workers del DataLoader.
    pin_memory : bool
        Recomendado True si entrenas en GPU.
    drop_last : bool
        Si True, descarta el último batch incompleto.
    shuffle_train : bool
        Si True, baraja el set de entrenamiento.

    Returns
    -------
    train_loader, val_loader, test_loader : torch.utils.data.DataLoader
    """

    # -------------------- Validación split --------------------
    if not (isinstance(split, (tuple, list)) and len(split) == 3):
        raise ValueError("split debe ser una tupla/lista de 3 elementos: (train, val, test).")

    tr, va, te = map(float, split)
    if tr <= 0 or va < 0 or te < 0:
        raise ValueError("split debe tener proporciones no negativas y train > 0.")
    if abs((tr + va + te) - 1.0) > 1e-9:
        raise ValueError("split debe sumar 1.0 (ej: (0.8, 0.1, 0.1)).")

    # -------------------- Convertir a tensores --------------------
    if isinstance(X, np.ndarray):
        X_t = torch.from_numpy(X)
    elif torch.is_tensor(X):
        X_t = X
    else:
        raise TypeError("X debe ser np.ndarray o torch.Tensor.")

    if isinstance(y, np.ndarray):
        y_t = torch.from_numpy(y)
    elif torch.is_tensor(y):
        y_t = y
    else:
        raise TypeError("y debe ser np.ndarray o torch.Tensor.")

    if y_t.ndim != 1:
        raise ValueError("y debe ser un vector 1D con shape (N,).")

    N = X_t.shape[0]
    if y_t.shape[0] != N:
        raise ValueError(f"Dimensión inconsistente: X tiene N={N} pero y tiene {y_t.shape[0]}.")

    # Asegurar dtype de etiquetas
    if y_t.dtype not in (torch.int64, torch.long):
        y_t = y_t.long()

    dataset = TensorDataset(X_t, y_t)

    # -------------------- Split estratificado --------------------
    y_np = y_t.cpu().numpy()
    rng = np.random.RandomState(seed)

    train_idx, val_idx, test_idx = [], [], []
    classes = np.unique(y_np)

    for c in classes:
        idx = np.where(y_np == c)[0]
        rng.shuffle(idx)

        n = len(idx)
        n_train = int(np.floor(tr * n))
        n_val = int(np.floor(va * n))
        # el resto a test para cuadrar por clase
        n_test = n - n_train - n_val

        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train:n_train + n_val])
        test_idx.append(idx[n_train + n_val:])

    train_idx = np.concatenate(train_idx) if len(train_idx) else np.array([], dtype=np.int64)
    val_idx = np.concatenate(val_idx) if len(val_idx) else np.array([], dtype=np.int64)
    test_idx = np.concatenate(test_idx) if len(test_idx) else np.array([], dtype=np.int64)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())
    test_ds = Subset(dataset, test_idx.tolist())

    # -------------------- Reproducibilidad DataLoader --------------------
    def seed_worker(worker_id):
        # Inicializa RNG por worker de forma determinista
        worker_seed = (torch.initial_seed() + worker_id) % 2**32
        np.random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    # -------------------- DataLoaders --------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=g if shuffle_train else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
    )

    return train_loader, val_loader, test_loader
