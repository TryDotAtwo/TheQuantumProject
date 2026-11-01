from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class QuantumFeatureResult:
    features: np.ndarray
    mode: str
    cache_hit: bool


class LRUCache:
    def __init__(self, max_size: int = 100000):
        if max_size < 0:
            raise ValueError("cache size must be nonnegative")
        self.max_size = int(max_size)
        self._data: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, key: str) -> Optional[np.ndarray]:
        if key not in self._data:
            return None
        value = self._data.pop(key)
        self._data[key] = value
        return value

    def put(self, key: str, value: np.ndarray) -> None:
        if self.max_size == 0:
            return
        self._data.pop(key, None)
        self._data[key] = value
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)


class QuantumFeatureEmulator:
    """Optional Qiskit Aer feature circuit; requested backends never silently downgrade."""

    def __init__(self, n_qubits: int = 4, n_layers: int = 2, mode: str = "aer_cpu", cache_size: int = 100000):
        if n_qubits <= 0 or n_layers < 0:
            raise ValueError("n_qubits must be positive and n_layers nonnegative")
        if mode not in {"none", "aer_cpu", "aer_gpu"}:
            raise ValueError(f"Unsupported quantum mode: {mode}")
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.mode = mode
        self.cache = LRUCache(cache_size)
        self.backend = None
        self._circuit_type = None
        self._transpile = None

        if mode == "none":
            return
        try:
            from qiskit import QuantumCircuit, transpile
            from qiskit_aer import AerSimulator
        except ImportError as exc:
            raise RuntimeError("Qiskit Aer mode was requested, but qiskit and qiskit-aer are unavailable.") from exc

        device = "GPU" if mode == "aer_gpu" else "CPU"
        try:
            self.backend = AerSimulator(method="statevector", device=device)
        except Exception as exc:
            raise RuntimeError(f"Could not initialize the requested Aer {device} backend; no fallback was attempted.") from exc
        self._circuit_type = QuantumCircuit
        self._transpile = transpile

    def is_available(self) -> bool:
        return self.backend is not None

    def _key(self, x: np.ndarray) -> str:
        return hashlib.sha256(np.asarray(x, dtype=np.float32).tobytes()).hexdigest()

    def _build_circuit(self, x: np.ndarray):
        if self.backend is None or self._circuit_type is None:
            raise RuntimeError("No quantum backend is configured; choose aer_cpu or aer_gpu.")
        qc = self._circuit_type(self.n_qubits)
        for qubit in range(self.n_qubits):
            angle_y = float(x[qubit % len(x)]) * np.pi
            angle_z = float(x[(qubit + self.n_qubits) % len(x)]) * np.pi
            qc.ry(angle_y, qubit)
            qc.rz(angle_z, qubit)
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits - 1):
                qc.cx(qubit, qubit + 1)
            if self.n_qubits > 1:
                qc.cx(self.n_qubits - 1, 0)
            for qubit in range(self.n_qubits):
                angle_y = float(x[(layer + qubit) % len(x)]) * np.pi
                angle_z = float(x[(layer + qubit + 3) % len(x)]) * np.pi
                qc.ry(angle_y, qubit)
                qc.rz(angle_z, qubit)
        qc.save_statevector()
        return qc

    def _expectations_from_statevector(self, statevector: np.ndarray) -> np.ndarray:
        probabilities = np.abs(statevector) ** 2
        expectations = np.zeros(self.n_qubits, dtype=np.float32)
        for qubit in range(self.n_qubits):
            bit_values = np.fromiter(
                (1.0 if (basis_state >> qubit) & 1 == 0 else -1.0 for basis_state in range(len(probabilities))),
                dtype=np.float32,
                count=len(probabilities),
            )
            expectations[qubit] = np.dot(probabilities, bit_values)
        return np.concatenate((expectations, expectations**2)).astype(np.float32, copy=False)

    def compute(self, x: np.ndarray) -> QuantumFeatureResult:
        values = np.asarray(x, dtype=np.float32)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"quantum input must be a nonempty vector, got shape {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("quantum input contains NaN or infinite values")

        key = self._key(values)
        cached = self.cache.get(key)
        if cached is not None:
            return QuantumFeatureResult(features=cached.copy(), mode=self.mode, cache_hit=True)
        if self.backend is None or self._transpile is None:
            raise RuntimeError("Quantum backend unavailable. Install qiskit-aer or choose quantum_mode='none'.")

        circuit = self._build_circuit(values)
        compiled = self._transpile(circuit, self.backend)
        result = self.backend.run(compiled).result()
        if not result.success:
            raise RuntimeError(f"Aer {self.mode} simulation failed: {result.status}")
        statevector = np.asarray(result.data(0)["statevector"], dtype=np.complex128)
        features = self._expectations_from_statevector(statevector)
        self.cache.put(key, features)
        return QuantumFeatureResult(features=features.copy(), mode=self.mode, cache_hit=False)
