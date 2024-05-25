import pygame
import numpy as np

# Initialize Pygame
pygame.init()
width, height = 800, 400
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Function to generate heatmap data
def generate_heatmap_data():
    return np.random.randint(0, 256, (100, 100), dtype=np.uint8)

# Function to manually create a red heatmap surface with alpha matching the data values
def create_red_heatmap_surface(heatmap_data):
    # Create a surface capable of handling alpha
    heatmap_surface = pygame.Surface(heatmap_data.shape, pygame.SRCALPHA)
    pix_array = pygame.PixelArray(heatmap_surface)

    for x in range(100):
        for y in range(100):
            value = heatmap_data[x, y]
            # Set pixel value; format: (Red, Green, Blue, Alpha)
            pix_array[x, y] = (value, 0, 0, value)

    # Delete the pixel array to unlock the surface
    del pix_array

    # Scale the surface to the window size
    return pygame.transform.scale(heatmap_surface, (width, height))

running = True
while running:
    screen.fill((255,255,255))
    for event in pygame.event.get():
        if event.type is pygame.QUIT:
            running = False

    # Generate new heatmap data
    heatmap_data = generate_heatmap_data()

    # Update the heatmap display
    heatmap_surface = create_red_heatmap_surface(heatmap_data)
    screen.blit(heatmap_surface, (0, 0))

    pygame.display.flip()
    clock.tick(60)  # Cap at 60 frames per second

pygame.quit()
