import numpy as np
from pygame import draw, Rect

grid_color = (128, 128, 128)
background_color = (0, 0, 0)


class GameHandler:
    def __init__(self, dim: (float, float), pos: (float, float), grid_size: int, game_id):
        self.dim = np.array(dim)
        self.pos = np.array(pos)
        self.grid_size = grid_size
        self.grid_spacing = np.array([dim[0]/grid_size, dim[1]/grid_size])
        self.game_id = game_id

        self.is_running = True
        print(self.game_id)

    def draw(self, surface):
        rect = Rect(self.pos[0], self.pos[1], self.dim[0], self.dim[1])
        draw.rect(surface, background_color, rect)
        for i in range(self.grid_size+1):
            draw.line(surface, grid_color,
                      (self.pos[0], self.pos[1]+i*self.grid_spacing[1]),
                      (self.pos[0]+self.dim[0], self.pos[1]+i*self.grid_spacing[1]))
            draw.line(surface, grid_color,
                      (self.pos[0] + i * self.grid_spacing[0], self.pos[1]),
                      (self.pos[0] + i * self.grid_spacing[0], self.pos[1] + self.dim[1]))

    def step(self):
        if self.is_running:
            handle_death_conditions()

    def handle_death_conditions(self):
        pass
