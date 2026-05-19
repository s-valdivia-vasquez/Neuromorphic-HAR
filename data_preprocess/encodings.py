# sigma_delta_coding.py  (Python 3.7+)
from typing import Dict, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd

def center_per_channel(x: np.ndarray) -> np.ndarray:
    """Resta la media temporal por canal (útil para quitar DC/gravedad)."""
    return x - x.mean(axis=0, keepdims=True)


def default_step_from_std(x: np.ndarray, k: float = 0.1, eps: float = 1e-6) -> np.ndarray:
    """
    Heurística simple para elegir theta/delta por canal:
        step[c] = max(eps, k * std[c])
    """
    s = x.std(axis=0)
    return np.maximum(eps, k * s).astype(np.float32)

def sigma_delta(
    x: np.ndarray,
    theta: Union[float, np.ndarray],
    init: str = "x0",
    return_reconstruction: bool = True,
    dead_zone: Optional[Union[float, np.ndarray]] = None,
    token: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Optional[np.ndarray]]:
    """
    Implementa ΣΔ encoding con dead-zone opcional y token inicial opcional.

    Parámetros
    ----------
    x : np.ndarray
        Señal contigua [T,C].
    theta : float o np.ndarray [C]
        Paso de reconstrucción por canal.
    init : str
        "x0"   -> xhat inicial = x[0]
        "zero" -> xhat inicial = 0
    return_reconstruction : bool
        Si True, retorna la reconstrucción temporal.
    dead_zone : None, float o np.ndarray [C]
        - None  -> sin dead-zone
        - float -> dead-zone relativa: |dx| > dead_zone * theta
        - array -> dead-zone absoluta por canal
    token : bool
        Si True, guarda x[0,c] en events[0,c,0] y la reconstrucción parte desde x[0].

    Retorna
    -------
    events : np.ndarray
        [T,C,2] float32: (up, down).
        Con token=False, todos los eventos son 0/1.
        Con token=True, solo events[0,:,0] puede ser distinto de 0/1.
    state : dict
        {"xhat": xhat_last}
    xhat_trace : np.ndarray o None
        Reconstrucción [T,C] si return_reconstruction=True.
    """
    if x.ndim != 2:
        raise ValueError("x debe ser [T,C]")

    x = np.asarray(x, dtype=np.float32)
    T, C = x.shape

    # --- theta ---
    th = np.asarray(theta, dtype=np.float32)
    if th.ndim == 0:
        th = np.full((C,), float(th), dtype=np.float32)
    elif th.shape != (C,):
        raise ValueError("theta debe ser escalar o vector [C]")

    # --- dead zone ---
    if dead_zone is None:
        dz = np.zeros((C,), dtype=np.float32)
    else:
        dz = np.asarray(dead_zone, dtype=np.float32)
        if dz.ndim == 0:
            dz = dz * th
        elif dz.shape != (C,):
            raise ValueError("dead_zone debe ser None, escalar o vector [C]")

    # float32 para permitir que el token inicial no sea binario
    events = np.zeros((T, C, 2), dtype=np.float32)
    xhat_trace = np.empty((T, C), dtype=np.float32) if return_reconstruction else None

    # --- inicio ---
    if token:
        # el token inicial se guarda en el canal positivo
        events[0, :, 0] = x[0]
        xhat = x[0].copy()

        if xhat_trace is not None:
            xhat_trace[0] = xhat

        t0 = 1
    else:
        if init == "x0":
            xhat = x[0].copy()
        elif init == "zero":
            xhat = np.zeros((C,), dtype=np.float32)
        else:
            raise ValueError('init debe ser "x0" o "zero"')

        t0 = 0

    # --- bucle temporal ---
    for t in range(t0, T):
        dx = x[t] - xhat
        fire = np.abs(dx) > dz

        y = np.zeros((C,), dtype=np.int8)
        y[fire] = np.where(dx[fire] >= 0.0, 1, -1).astype(np.int8)

        events[t, :, 0] = (y == 1).astype(np.float32)
        events[t, :, 1] = (y == -1).astype(np.float32)

        xhat = xhat + y.astype(np.float32) * th

        if xhat_trace is not None:
            xhat_trace[t] = xhat

    state = {"xhat": xhat.copy()}
    return events, state, xhat_trace

def level_crossing(x: np.ndarray,delta: Union[float, np.ndarray],init: str = "x0",max_crossings_per_sample: int = 15,return_level_trace: bool = False) -> Tuple[np.ndarray, Dict[str, np.ndarray], Optional[np.ndarray]]:
    """
    Level-crossing / send-on-delta contiguo.

    A diferencia del ΣΔ del paper, aquí puedes tener múltiples cruces en una muestra
    (conteos up/down), lo que ayuda si hay saltos grandes entre muestras.
    """
    if x.ndim != 2:
        raise ValueError("x debe ser [T,C]")
    T, C = x.shape

    dv = np.asarray(delta, dtype=np.float32)
    if dv.ndim == 0:
        dv = np.full((C,), float(dv), dtype=np.float32)
    elif dv.shape != (C,):
        raise ValueError("delta debe ser escalar o vector [C]")

    if init == "x0":
        level = x[0].astype(np.float32, copy=True)
    elif init == "zero":
        level = np.zeros((C,), dtype=np.float32)
    else:
        raise ValueError('init debe ser "x0" o "zero"')

    events = np.zeros((T, C, 2), dtype=np.uint8)
    level_trace = np.empty((T, C), dtype=np.float32) if return_level_trace else None

    for t in range(T):
        diff = x[t] - level

        up = np.floor(np.maximum(diff, 0.0) / dv)
        dn = np.floor(np.maximum(-diff, 0.0) / dv)

        if max_crossings_per_sample is not None:
            up = np.minimum(up, max_crossings_per_sample)
            dn = np.minimum(dn, max_crossings_per_sample)

        events[t, :, 0] = up.astype(np.uint8)
        events[t, :, 1] = dn.astype(np.uint8)

        level = level + (up - dn) * dv

        if level_trace is not None:
            level_trace[t] = level

    state = {"level": level.copy()}
    return events, state, level_trace

