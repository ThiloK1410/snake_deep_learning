import pygame
from game_handler import GameHandler

screen_size = (500, 500)
x_padding = 10
y_padding = 10
game_grid_size = (3, 3)

game_count = game_grid_size[0]*game_grid_size[1]


class App:
    # main function from where everything is called
    def __init__(self):
        # initiating a clock and setting timer of the application
        self.clock = pygame.time.Clock()
        self.fps = 30
        self.time_per_frame = 1000 / self.fps

        self._running = True
        self.display = None

        self.size = screen_size

        self.game_handlers = []

    # called once to start program
    def on_init(self):
        pygame.init()
        self.display = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self._running = True

        # calculating the dimensions of every game block
        game_dim = ((screen_size[0]-((game_grid_size[0]+1)*x_padding))/game_grid_size[0],
                    (screen_size[1]-((game_grid_size[1]+1)*y_padding))/game_grid_size[1])
        for i in range(game_grid_size[1]):
            for j in range(game_grid_size[0]):
                game_pos = ((x_padding+j*(game_dim[0]+x_padding)), (y_padding+i*(game_dim[1]+y_padding)))
                self.game_handlers.append(GameHandler(game_dim, game_pos, 10, j+i*game_grid_size[0]))

        self.on_execute()

    # handles player inputs
    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                pass

    # loop which will be executed at fixed rate (for physics, animations and such)
    def on_loop(self):
        pass

    # loop which will only be called when enough cpu time is available
    def on_render(self):
        self.display.fill((255, 255, 255))

        for game in self.game_handlers:
            game.draw(self.display)

        pygame.display.update()

    @staticmethod
    def on_cleanup():
        pygame.quit()

    def on_execute(self):


        previous = pygame.time.get_ticks()
        lag = 0.0

        # game loop
        while self._running:
            current = pygame.time.get_ticks()
            elapsed = current - previous
            lag += elapsed
            previous = current

            for event in pygame.event.get():
                self.on_event(event)

            while lag > self.time_per_frame:
                self.on_loop()
                lag -= self.time_per_frame
            self.on_render()
        self.on_cleanup()


if __name__ == "__main__":
    app = App()
    app.on_init()
