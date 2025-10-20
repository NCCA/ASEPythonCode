from dataclasses import dataclass

from Vec3 import Vec3


@dataclass
class Particle:
    position: Vec3
    direction: Vec3
    colour: Vec3
    max_life: int = 1000
    life: int = 0
    scale : float = 1.0

