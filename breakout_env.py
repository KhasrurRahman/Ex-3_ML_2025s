import random

class BreakoutEnv:
    def __init__(self, num_bricks=10, grid_width=15, grid_height=10):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.paddle_width = 5
        self.paddle_y = 0
        self.brick_width = 3
        self.brick_height = 1
        self.num_bricks = num_bricks
        self.reset()

    def reset(self, layout="line"):
        # Paddle in center
        self.paddle_x = (self.grid_width - self.paddle_width) // 2
        self.paddle_speed = 0
        # Ball in center, random direction
        self.ball_x = self.grid_width // 2
        self.ball_y = self.paddle_y + 2
        self.ball_vx, self.ball_vy = random.choice([(-2,1), (-1,1), (0,1), (1,1), (2,1)])
        # Bricks layout
        if layout == "line":
            self.bricks = [(i * self.brick_width, self.grid_height - 2) for i in range(self.num_bricks)]
        elif layout == "block":
            self.bricks = []
            rows = 2
            cols = self.num_bricks // rows
            for row in range(rows):
                for col in range(cols):
                    x = col * self.brick_width
                    y = self.grid_height - 2 - row
                    self.bricks.append((x, y))
        elif layout == "pyramid":
            self.bricks = []
            levels = 3
            start_y = self.grid_height - 2
            for level in range(levels):
                for i in range(level * 2 + 1):
                    x = (self.grid_width // 2) - level * self.brick_width + i * self.brick_width
                    y = start_y - level
                    self.bricks.append((x, y))
        elif layout == "random":
            self.bricks = []
            for _ in range(self.num_bricks):
                x = random.randint(0, self.grid_width - self.brick_width)
                y = random.randint(self.grid_height - 4, self.grid_height - 2)
                self.bricks.append((x, y))
        else:
            raise ValueError("Unsupported layout type")
        self.done = False
        return self._get_state()

    def step(self, action):
        # Paddle movement
        self.paddle_speed += action
        self.paddle_speed = max(-2, min(2, self.paddle_speed))
        self.paddle_x += self.paddle_speed
        self.paddle_x = max(0, min(self.grid_width - self.paddle_width, self.paddle_x))

        # Ball movement
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Ball-wall bounce
        if self.ball_x < 0 or self.ball_x >= self.grid_width:
            self.ball_vx *= -1
            self.ball_x = max(0, min(self.grid_width - 1, self.ball_x))
        if self.ball_y >= self.grid_height:
            self.ball_vy *= -1
            self.ball_y = self.grid_height - 1

        # Ball-paddle bounce
        if self.ball_y == self.paddle_y + 1 and self.paddle_x <= self.ball_x < self.paddle_x + self.paddle_width:
            hit_pos = self.ball_x - self.paddle_x
            self.ball_vx, self.ball_vy = [(-2,1), (-1,1), (0,1), (1,1), (2,1)][hit_pos * 5 // self.paddle_width]
            self.ball_y = self.paddle_y + 2

        # Ball-brick collision
        for bx, by in self.bricks:
            if bx <= self.ball_x < bx + self.brick_width and by == self.ball_y:
                self.bricks.remove((bx, by))
                self.ball_vy *= -1
                break

        # Ball missed: reset environment instead of ending
        if self.ball_y < 0:
            # Ball missed: reset environment instead of ending
            self.reset()
            return self._get_state(), -1, False

        # All bricks cleared
        if not self.bricks:
            self.done = True
            return self._get_state(), 100, True

        return self._get_state(), -1, False

    def _get_state(self):
        return (self.ball_x, self.ball_y, self.ball_vx, self.ball_vy, self.paddle_x, tuple(self.bricks))

    def render(self):
        grid = [[' ' for _ in range(self.grid_width)] for _ in range(self.grid_height)]

        # Draw bricks
        for bx, by in self.bricks:
            for i in range(self.brick_width):
                if 0 <= bx + i < self.grid_width:
                    grid[by][bx + i] = '#'

        # Draw paddle
        for i in range(self.paddle_width):
            if 0 <= self.paddle_x + i < self.grid_width:
                grid[self.paddle_y][self.paddle_x + i] = '='

        # Draw ball
        if 0 <= self.ball_y < self.grid_height and 0 <= self.ball_x < self.grid_width:
            grid[self.ball_y][self.ball_x] = 'O'

        # Print grid from top to bottom
        print("\n".join("".join(row) for row in reversed(grid)))
        print("-" * self.grid_width)
