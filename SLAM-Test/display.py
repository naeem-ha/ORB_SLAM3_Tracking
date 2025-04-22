# PyGame > SDL2 > OpenCV
import pygame
from pygame.locals import DOUBLEBUF
import cv2

class Display(object):
  def __init__(self, W, H):
    pygame.init()
    self.screen = pygame.display.set_mode((W, H), DOUBLEBUF)
    self.surface = pygame.Surface(self.screen.get_size()).convert()

  def display2D(self, img):
      if len(img.shape) == 2:  # grayscale
          img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

      pygame.surfarray.blit_array(self.surface, img.swapaxes(0,1)[:, :, [0,1,2]])
      self.screen.blit(self.surface, (0,0))
      pygame.display.flip()

