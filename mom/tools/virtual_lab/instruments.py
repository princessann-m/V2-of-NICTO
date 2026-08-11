"""Virtual instruments that produce real computational results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class VirtualInstrument(ABC):
    """Base virtual instrument."""

    name: str = "instrument"

    @abstractmethod
    def measure(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


class Thermometer(VirtualInstrument):
    """Virtual thermometer producing real temperature computations."""

    name = "thermometer"

    def measure(self, initial_temp: float, time: float, ambient: float, cooling_rate: float = 0.1) -> Dict[str, Any]:
        temp = ambient + (initial_temp - ambient) * np.exp(-cooling_rate * time)
        return {
            "type": "TEMPERATURE",
            "value": float(temp),
            "units": "C",
            "label": "VIRTUAL EXPERIMENT RESULT",
        }


class Voltmeter(VirtualInstrument):
    """Virtual voltmeter producing real circuit measurements."""

    name = "voltmeter"

    def measure(self, resistance: float, current: float) -> Dict[str, Any]:
        voltage = resistance * current
        return {
            "type": "VOLTAGE",
            "value": float(voltage),
            "units": "V",
            "label": "VIRTUAL EXPERIMENT RESULT",
        }


class Stopwatch(VirtualInstrument):
    """Virtual stopwatch producing real timing."""

    name = "stopwatch"

    def measure(self, distance: float, speed: float) -> Dict[str, Any]:
        if speed <= 0:
            raise ValueError("Speed must be positive")
        time = distance / speed
        return {
            "type": "TIME",
            "value": float(time),
            "units": "s",
            "label": "VIRTUAL EXPERIMENT RESULT",
        }


class Spectrometer(VirtualInstrument):
    """Virtual spectrometer producing real spectral data."""

    name = "spectrometer"

    def measure(self, wavelength_range: tuple[float, float], peaks: int = 3) -> Dict[str, Any]:
        wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], 1000)
        intensity = np.zeros_like(wavelengths)
        for i in range(peaks):
            center = np.random.uniform(wavelength_range[0] + 50, wavelength_range[1] - 50)
            width = np.random.uniform(5, 20)
            amplitude = np.random.uniform(0.3, 1.0)
            intensity += amplitude * np.exp(-((wavelengths - center) ** 2) / (2 * width**2))
        intensity += np.random.normal(0, 0.02, size=wavelengths.shape)
        intensity = np.clip(intensity, 0, None)
        return {
            "type": "SPECTRUM",
            "wavelengths": wavelengths.tolist(),
            "intensity": intensity.tolist(),
            "label": "VIRTUAL EXPERIMENT RESULT",
        }
