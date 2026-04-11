from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    HAVE_QISKIT_AER = True
except Exception:
    HAVE_QISKIT_AER = False


@dataclass
class QuantumFeatureResult:
    features: np.ndarray
    mode: str
    cache_hit: bool


class LRUCache:
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self._data = OrderedDict()

    def get(self, key):
        if key not in self._data:
            return None
        value = self._data.pop(key)
        self._data[key] = value
        return value

    def put(self, key, value):
        if key in self._data:
            self._data.pop(key)
        self._data[key] = value
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)


class QuantumFeatureEmulator:
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, mode: str = "aer_cpu", cache_size: int = 100000):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.mode = mode
        self.cache = LRUCache(cache_size)
        self.backend = None
        if HAVE_QISKIT_AER and mode in {"aer_cpu", "aer_gpu"}:
            device = "GPU" if mode == "aer_gpu" else "CPU"
            try:
                self.backend = AerSimulator(method="statevector", device=device)
            except Exception:
                self.backend = AerSimulator(method="statevector")
                self.mode = "aer_cpu"

    def is_available(self) -> bool:
        return self.backend is not None

    def _key(self, x: np.ndarray) -> str:
        return hashlib.sha1(np.asarray(x, dtype=np.float32).tobytes()).hexdigest()

    def _build_circuit(self, x: np.ndarray):
        if not HAVE_QISKIT_AER:
            raise RuntimeError("Qiskit Aer is not installed.")
        x = np.asarray(x, dtype=np.float32)
        qc = QuantumCircuit(self.n_qubits)
        # angle encoding
        for q in range(self.n_qubits):
            a = float(x[q % len(x)]) * np.pi
            b = float(x[(q + self.n_qubits) % len(x)]) * np.pi
            qc.ry(a, q)
            qc.rz(b, q)
        for layer in range(self.n_layers):
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
            qc.cx(self.n_qubits - 1, 0)
            for q in range(self.n_qubits):
                a = float(x[(layer + q) % len(x)]) * np.pi
                b = float(x[(layer + q + 3) % len(x)]) * np.pi
                qc.ry(a, q)
                qc.rz(b, q)
        qc.save_statevector()
        return qc

    def _expectations_from_statevector(self, statevector: np.ndarray) -> np.ndarray:
        n = self.n_qubits
        probs = np.abs(statevector) ** 2
        out = np.zeros(2 * n, dtype=np.float32)
        for q in range(n):
            z_exp = 0.0
            for idx, p in enumerate(probs):
                bit = (idx >> q) & 1
                z_exp += p * (1.0 if bit == 0 else -1.0)
            out[q] = z_exp
        # simple quadratic features
        out[n:] = out[:n] ** 2
        return out

    def compute(self, x: np.ndarray) -> QuantumFeatureResult:
        key = self._key(x)
        cached = self.cache.get(key)
        if cached is not None:
            return QuantumFeatureResult(features=cached.copy(), mode=self.mode, cache_hit=True)
        if self.backend is None:
            raise RuntimeError("Quantum backend unavailable. Install qiskit-aer or choose quantum_mode='none'.")
        qc = self._build_circuit(x)
        tqc = transpile(qc, self.backend)
        result = self.backend.run(tqc).result()
        state = np.asarray(result.data(0)["statevector"], dtype=np.complex128)
        feats = self._expectations_from_statevector(state)
        self.cache.put(key, feats)
        return QuantumFeatureResult(features=feats, mode=self.mode, cache_hit=False)
