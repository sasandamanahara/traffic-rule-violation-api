"""
Live detection modules for real-time violation detection
"""
from .helmet_triple_detector import HelmetTripleDetector
from .speed_detector import SpeedDetector
from .direction_detector import DirectionDetector
from .redlight_detector import RedLightDetector
from .noparking_detector import NoParkingDetector

__all__ = [
    'HelmetTripleDetector',
    'SpeedDetector',
    'DirectionDetector',
    'RedLightDetector',
    'NoParkingDetector'
]


