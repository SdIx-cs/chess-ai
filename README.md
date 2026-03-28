# AI Chess Agent

AI Chess Agent is a Python-based AI chess framework built on [ChessMaker] (https://pypi.org/project/chessmaker/) for game simulation. It implements Minimax with Alpha-Beta pruning, iterative deepening, transposition tables, and killer moves, with configurable depth and piece weights. Supporting Opening & Endgame modes, the project demonstrates AI strategy, performance optimization, and algorithmic problem-solving in a full-featured chess engine.

---

## Tech Stack

* **Language:** Python 3.11.0
* **Libraries:** ChessMaker (chess simulation), standard Python libraries
* **Environment:** Conda or native Python virtual environment

---

## Features

* **Minimax Algorithm** – Evaluates moves with configurable depth (ply) for strategic foresight. Implemented recursively with attention to efficiency.
* **Alpha-Beta Pruning** – Optimizes search by pruning branches that cannot improve the AI’s evaluation.
* **Piece Weights** – Dynamic evaluation of pieces guides decision-making based on board stage.
Transposition Table – Caches board positions for fast lookup, tracking hits and move history to reduce redundant computations.
* **Pruning Techniques** – Combines table hits and heuristics to skip irrelevant branches, speeding up decisions.
* **Game Modes** – Supports Opening and Endgame modes, with AI adapting strategies to the current board configuration.

---

## Environment Setup

### Option 1: Using Conda (Recommended)

```bash
conda create -n chess_ai python=3.12.9
conda activate chess_ai
pip install chessmaker
```

### Option 2: Without Conda

Ensure your environment includes:

* Python ≥ 3.11 (3.12.9 preferred for compatibility)
* ChessMaker: `pip install chessmaker`

Finally, clone this repository:

```bash
git clone <repository-url>
cd chess_ai
```

---

## Project Structure

```plaintext
<chess-ai>
├── extension/
│   ├── board_rules.py
│   ├── board_utils.py
│   ├── piece_pawn.py
│   └── piece_right.py
├── agent.py
├── opponent1.py
├── opponent2.py
├── samples.py
├── test.py
└── README.md
```

* **extension/** – Utility and rule files (read-only; see Notes).
* **agent.py** – The AI agent implementation.
* **opponent1.py** – Example opponent AI 1: Baseline random player.
* **opponent2.py** – Example opponent AI 2: Stronger, more strategic player.
* **samples.py** – Board configurations.
* **test.py** – Script to run a game simulation.

---

## Running a Test Game

1. Navigate to the repository:

```bash
cd chess-ai
```

2. Modify `test.py` to configure players and board setups:

*Switching Opponents*
```python
opponent = opponent2   # change to opponent1 for baseline opponent
```

*Switching Player Sides*
```python
if __name__ == "__main__":
    testgame(p_white=agent, p_black=opponent.opponent, board_sample=sample0)
```

* Agent as **White**:

  ```python
  testgame(p_white=agent, p_black=opponent.opponent, board_sample=sample0)
  ```

* Agent as **Black**:

  ```python
  testgame(p_white=opponent.opponent, p_black=agent, board_sample=sample0)
  ```

3. Run the game:

```bash
python test.py
```

4. Game outcomes will display in the terminal:

```plaintext
=== Game ended: Checkmate - black loses ===
```

5. To interrupt a running game:

```bash
Ctrl + C
=== Game ended by keyboard interruption ===
```

---

## Testing & Opponents

Baseline Opponent (opponent1): Makes random moves. The agent defeats it easily, confirming correctness and basic strategy implementation.
Strategic Opponent (opponent2): One of my earlier version of agent that plays strategically. While the current agent defeats opponent2 overall, it exposes weaknesses in endgame play, even with endgame heuristics implemented.

This highlights the importance of incrementally testing against stronger opponents to identify edge cases. 


---

## Notes

* `agent.py` and `opponent2.py` is my original AI implementation code. 
* `opponent1.py`, `samples.py` and the `extension/` directory contains **utility and example baseline code provided by the University of Southampton**, with minor modifications made to support custom opponent switching interactions. 
* The chess rules implemented in this agent are **slightly different from standard chess**, requiring the AI to adapt to unique game mechanics.

---

## Customizing The Agent

* **agent.py:** Implement your custom AI agent here.
* **opponent.py:** Example opponent; feel free to create a stronger AI.

**Important:** Only modify player selection and initial boards in `test.py`. Game logic in `extension/` should remain intact.

---

## Resources

* **ChessMaker Documentation:** [https://wolfdwyc.github.io/ChessMaker](https://wolfdwyc.github.io/ChessMaker)
* **Source Code:** [https://github.com/WolfDWyc/ChessMaker](https://github.com/WolfDWyc/ChessMaker)

---

**Key Skills Highlighted:** AI agent design, Python programming, game simulation, modular code structure, optimization with advanced search algorithms.

