import os.path
import numpy as np
from ctypes import CDLL, POINTER, c_int, c_double, c_void_p, Structure, sizeof
import pygame
import pygame.gfxdraw
import time

# Parameters
particle_count = 1500
color = pygame.Color(255,255,255,64)

# Launch pygame
pygame.init()
width, height = 1920, 1080
flags = pygame.HWSURFACE | pygame.DOUBLEBUF
screen = pygame.display.set_mode((width, height), flags, vsync=1)
pygame.display.set_caption("n-body")
clock = pygame.time.Clock()

# The particles type
class cParticles(Structure):
	_fields_ = [("xPos", POINTER(c_double)),
				("yPos", POINTER(c_double)),
				("xVelo", POINTER(c_double)),
				("yVelo", POINTER(c_double)),
				("mass", POINTER(c_double)),
				("n", c_double)]

# Load the dll
cwd = os.path.dirname(__file__)
many_body_dll = os.path.join(cwd, "many_body.dll")
shared_dll = CDLL(many_body_dll)

# Set up functions for use
prepare = shared_dll.prepare
prepare.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int]
prepare.restype = POINTER(cParticles)
update = shared_dll.update
update.argtypes = [c_double]
update.restype = POINTER(cParticles)
clean_up = shared_dll.cleanUp
clean_up.argtypes = []
clean_up.restype = c_int
free_array = shared_dll.freeArray
free_array.argtypes = [POINTER(cParticles)]

# Generate somes values
x_pos = (800*np.random.rand(particle_count)-400).ctypes.data_as(POINTER(c_double))
y_pos = (600*np.random.rand(particle_count)-300).ctypes.data_as(POINTER(c_double))
x_velo = np.zeros(particle_count).ctypes.data_as(POINTER(c_double))
y_velo = np.zeros(particle_count).ctypes.data_as(POINTER(c_double))
masses = (5e11*np.ones(particle_count)).ctypes.data_as(POINTER(c_double)) # 5e13*np.random.rand(particle_count)+

# Prepare compute shader
prepare(x_pos, y_pos, x_velo, y_velo, masses, particle_count)

# Main loop
delta_time = 0.02
running = True
while running:

	start_time = time.time()

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False
		elif event.type == pygame.KEYDOWN:
			if event.key == pygame.K_ESCAPE:
				running = False

	particles_ptr = update(c_double(delta_time))

	screen.fill((0,0,0))

	particles = particles_ptr.contents

	particle_surface = pygame.Surface((2, 2), pygame.SRCALPHA)
	pygame.draw.circle(particle_surface, (255, 255, 255, 64), (1, 1), 1)

	# In de render-loop:
	for i in range(particle_count):
		x = int(particles.xPos[i]) + width / 2
		y = int(particles.yPos[i]) + height / 2
		screen.blit(particle_surface, (x, y))

	pygame.display.flip()
	clock.tick(75)

	delta_time = time.time() - start_time

# Final cleanup
clean_up()
free_array(particles_ptr)
pygame.quit()
quit()