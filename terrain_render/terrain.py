from PIL import Image
import noise
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import os
from typing import Tuple
import sys

shape = (400, 400)

def generate_world(shape: Tuple[int, int] = (400, 400), type: str = "sine") -> np.ndarray:
    world = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if type == "sine":
                world[i][j] = - math.sin(i / (10*math.pi)) - math.cos(j / (10*math.pi))
            elif type == "perlin":  # Plot with Perlin Noise
                scale = 100.0
                octaves = 6
                persistence = 0.5
                lacunarity = 2.0
                world[i][j] = noise.pnoise2(i/scale, 
                    j/scale, 
                    octaves=octaves, 
                    persistence=persistence, 
                    lacunarity=lacunarity, 
                    repeatx=1024, 
                    repeaty=1024, 
                    base=42)
        
# world is a 400x400 vector with sine wave shape
    print(world.shape)
    return world

def draw_3d():
    world = generate_world(type="sine")
    lin_x = np.linspace(0, 1, shape[0], endpoint=False)
    lin_y = np.linspace(0, 1, shape[1], endpoint=False)
    x,y = np.meshgrid(lin_x, lin_y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, world, cmap='terrain')
    plt.show()

def output_grey(type="sine", difficulty_level: int = 1):
    if difficulty_level == 1:
        multiplier = 0.3
    elif difficulty_level == 2:
        multiplier = 0.6
    elif difficulty_level == 3:
        multiplier = 0.9
    else:
        multiplier = 0.3
    world = generate_world(type=type)
    world_normalized = (world - world.min()) / (world.max() - world.min())
    output = Image.fromarray(multiplier * world_normalized * 255).convert("RGBA")
    output.save("output_" + type + "_lv" + str(difficulty_level) + ".png")
    # plt.gray()
    # plt.gca().set_axis_off()
    # plt.imshow(world)
    # plt.savefig("output_" + type + ".png", transparent=True)

if __name__ == '__main__':
    output_grey(type=sys.argv[1], difficulty_level=int(sys.argv[2]))
    
