import numpy as np
from pygame import draw, Rect

grid_color = (128, 128, 128)
background_color = (0, 0, 0)
cell_color = (0, 128, 0)
food_color = (255, 204, 0)

# dqn parameters
step_count = 100
food_step_bonus = 50
allow_self_kill_through_movement = True

directions = {
    0: (1, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1)
}


class GameHandler:
    def __init__(self, dim: (float, float), pos: (float, float), grid_size: int, game_id):
        self.dim = np.array(dim)
        self.pos = np.array(pos)
        self.grid_size = grid_size
        self.grid_spacing = np.array([dim[0]/grid_size, dim[1]/grid_size])
        self.game_id = game_id

        self.is_running = True
        self.cells = []
        self.food_pos: (int, int) = None
        self.is_growing = False
        self.direction = 0
        self.game_init()

    def draw(self, surface):
        if self.is_running:
            rect = Rect(self.pos[0], self.pos[1], self.dim[0], self.dim[1])
            draw.rect(surface, background_color, rect)

            # drawing cells
            for cell in self.cells:
                rect = Rect(self.pos[0]+cell[0]*self.grid_spacing[0], self.pos[1]+cell[1]*self.grid_spacing[1],
                            self.grid_spacing[0], self.grid_spacing[1])
                draw.rect(surface, cell_color, rect)

            # drawing food
            rect = Rect(self.pos[0] + self.food_pos[0] * self.grid_spacing[0],
                        self.pos[1] + self.food_pos[1] * self.grid_spacing[1],
                        self.grid_spacing[0], self.grid_spacing[1])
            draw.rect(surface, food_color, rect)

            # drawing grid
            for i in range(self.grid_size+1):
                draw.line(surface, grid_color,
                          (self.pos[0], self.pos[1]+i*self.grid_spacing[1]),
                          (self.pos[0]+self.dim[0], self.pos[1]+i*self.grid_spacing[1]))
                draw.line(surface, grid_color,
                          (self.pos[0] + i * self.grid_spacing[0], self.pos[1]),
                          (self.pos[0] + i * self.grid_spacing[0], self.pos[1] + self.dim[1]))

    def step(self, action):
        if self.is_running:
            new_state, reward, terminated = (None, 0, False)
            head_pos = self.cells[0]
            if not self.is_growing:
                self.cells.pop()
            else:
                self.is_growing = False

            self.cells = [(head_pos[0]+directions[action][0], head_pos[1]+directions[action][1])] + self.cells

            if self.handle_food_collisions():
                reward = 1
            if self.handle_death_conditions():
                reward = -1
                terminated = True

            new_state = self.get_state()

            return new_state, reward, terminated

    def game_init(self):
        # spawning snake
        mid = self.grid_size//2
        self.cells = [(mid, mid), (mid-1, mid), (mid-2, mid)]
        self.is_running = True

        # spawning food
        self.spawn_food()

    def handle_death_conditions(self):
        head_pos = self.cells[0]
        if head_pos[0] < 0 or head_pos[0] >= self.grid_size or head_pos[1] < 0 or head_pos[1] >= self.grid_size:
            self.is_running = False
            return True

        for cell in self.cells[1:]:
            if head_pos[0] == cell[0] and head_pos[1] == cell[1]:
                self.is_running = False
                return True

        return False

    def handle_food_collisions(self):
        if self.food_pos[0] == self.cells[0][0] and self.food_pos[1] == self.cells[0][1]:
            self.grow_snake()
            self.spawn_food()
            return True
        return False

    def change_direction(self, dir_idx: int):
        if self.is_running and (dir_idx % 2 != self.direction % 2):
            self.direction = dir_idx

    def overlaps_with_snake(self, pos):
        for cell in self.cells:
            if cell[0] == pos[0] and cell[1] == pos[1]:
                return True
        return False

    def spawn_food(self):
        random_pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
        while self.overlaps_with_snake(random_pos):
            random_pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
        self.food_pos = random_pos

    def grow_snake(self):
        self.is_growing = True

    def get_state(self):
        food_offset_x = (self.food_pos[0] - self.cells[0][0]) / self.grid_size
        food_offset_y = (self.food_pos[1] - self.cells[0][1]) / self.grid_size
        is_right = float(self.direction==0)
        is_down = float(self.direction==1)
        is_left = float(self.direction==2)
        is_up = float(self.direction==3)
        danger_right = 0.
        danger_down = 0.
        danger_left = 0.
        danger_up = 0.
        head_pos_x = self.cells[0][0] / self.grid_size
        head_pos_y = self.cells[0][1] / self.grid_size

        head_pos = self.cells[0]

        for i in range(self.grid_size):
            pos = (head_pos[0] + (i+1), head_pos[1])
            if self.overlaps_with_snake(pos):
                danger_right = 1 - (pos[0]-head_pos[0]) / self.grid_size
                break

        for i in range(self.grid_size):
            pos = (head_pos[0], head_pos[1] + (i+1))
            if self.overlaps_with_snake(pos):
                danger_down = 1 - (pos[1]-head_pos[1]) / self.grid_size
                break

        for i in range(self.grid_size):
            pos = (head_pos[0] - (i+1), head_pos[1])
            if self.overlaps_with_snake(pos):
                danger_left = 1 - (pos[0]-head_pos[0]) / self.grid_size
                break

        for i in range(self.grid_size):
            pos = (head_pos[0], head_pos[1] - (i+1))
            if self.overlaps_with_snake(pos):
                danger_up = 1 - (pos[1]-head_pos[1]) / self.grid_size
                break


        return np.array([food_offset_x, food_offset_y,
                         is_right, is_down, is_left, is_up,
                         danger_right, danger_down, danger_left, danger_up,
                         head_pos_x, head_pos_y])

    @classmethod
    def get_state_size(cls):
        return 12

    @classmethod
    def get_action_space(cls):
        return [0, 1, 2, 3]
