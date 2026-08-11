"""Virtual laboratory package."""

from .lab import VirtualLab
from .instruments import VirtualInstrument, Thermometer, Voltmeter, Stopwatch, Spectrometer
from .experiments import Experiment, ExperimentStage, ExperimentRegistry

__all__ = [
    "VirtualLab",
    "VirtualInstrument",
    "Thermometer",
    "Voltmeter",
    "Stopwatch",
    "Spectrometer",
    "Experiment",
    "ExperimentStage",
    "ExperimentRegistry",
]
