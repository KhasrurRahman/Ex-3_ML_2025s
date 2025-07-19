Run code By:
```
python main.py
```


# Breakout RL – ML Exercise 3.3 (2025S)

Minimal grid-based simulation of the Breakout game used for Reinforcement Learning (Monte Carlo control).

## Files
- `breakout_env.py`: Game environment logic
- `main.py`: Runs the game and visualizes it
- `readMe.md`: This guide

## Setup Instructions

```bash
python3 -m venv venv
source venv/bin/activate      # On macOS/Linux
pip install matplotlib
```

## Run the Game

```bash
python main.py
```

- Plays with random paddle actions
- Visualizes each step using `matplotlib`
- Stops after 200 steps or if all bricks are cleared

## Layout Types

To change brick layouts, edit this line in `main.py`:

```python
state = env.reset(layout="pyramid")
```

Available layout options:
- `"line"` – horizontal row
- `"block"` – 2-row rectangle
- `"pyramid"` – triangle shape
- `"random"` – scattered bricks

## Game Info

- Grid size: 15 × 10
- Paddle: 5 blocks wide, max speed ±2
- Ball: 1×1, moves down with variable dx
- Action: -1 (left), 0 (stay), +1 (right)
- Reward: -1 per step, +100 when all bricks are cleared
- Missed ball: auto reset (bricks reappear, paddle recenters)

## Next Step

Train a Monte Carlo RL agent using this environment.