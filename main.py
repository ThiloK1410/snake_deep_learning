import pygame

class App:
    # main function from where everything is called
    def __init__(self):
        # initiating a clock and setting timer of the application
        self.clock = pygame.time.Clock()
        self.fps = 30
        self.time_per_frame = 1000 / self.fps

        self._running = True
        self.display = None

        self.size = (500, 500)

    # called once to start program
    def on_init(self):
        pygame.init()
        self.display = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self._running = True

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

        pygame.display.update()

    @staticmethod
    def on_cleanup():
        pygame.quit()

    def on_execute(self):
        previous = pygame.time.get_ticks()
        lag = 0.0

        # advanced game loop to call on_loop() at fixed rate and on_render() as fast as possible
        # (kinda overkill right now) (also not relevant)
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