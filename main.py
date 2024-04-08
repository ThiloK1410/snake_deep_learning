import pygame
from game_handler import GameHandler
from snake_dqn import SnakeDQL

screen_size = (800, 800)
x_padding = 10
y_padding = 10
game_grid_size = (3, 3)
game_size = 10

game_count = game_grid_size[0]*game_grid_size[1]


class App:
    # main function from where everything is called
    def __init__(self):
        # initiating a clock and setting timer of the application
        self.clock = pygame.time.Clock()
        self.fps = 3
        self.time_per_frame = 1000 / self.fps

        self._running = True
        self.display = None

        self.size = screen_size

        self.reinforcement_learner = SnakeDQL()

        self.episode_in_progress = False
        self.current_episode = 0

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
                self.game_handlers.append(GameHandler(game_dim, game_pos, game_size, j+i*game_grid_size[0]))

        self.display.fill((255, 255, 255))

        self.on_execute()

    # handles player inputs
    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_PLUS:
                self.fps += 1
            if event.key == pygame.K_MINUS:
                self.fps -= 1

    # loop which will be executed at fixed rate (for physics, animations and such)
    def on_loop(self):
        self.episode_in_progress = False
        for game in self.game_handlers:
            if game.is_running:
                self.episode_in_progress = True
                memory = game.step(self.reinforcement_learner(game.get_state()))
                self.reinforcement_learner.memorize(memory)

        # if all games are finished, train ai, reset them and start new episode
        if not self.episode_in_progress:
            self.reinforcement_learner.train()
            for game in self.game_handlers:
                game.game_init()
            self.episode_in_progress = True
            self.current_episode += 1
            print(f"CURRENT EPISODE: {self.current_episode}")

    # loop which will only be called when enough cpu time is available
    def on_render(self):

        for game in self.game_handlers:
            game.draw(self.display)

        pygame.display.update()

    @staticmethod
    def on_cleanup():
        pygame.quit()

    def on_execute(self):

        clock = pygame.time.Clock()
        previous = pygame.time.get_ticks()
        lag = 0.0

        # game loop
        while self._running:
            for event in pygame.event.get():
                self.on_event(event)

            self.on_loop()
            self.on_render()

            clock.tick(self.fps)
        self.on_cleanup()


if __name__ == "__main__":
    app = App()
    app.on_init()
