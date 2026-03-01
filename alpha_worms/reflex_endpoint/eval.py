import numpy as np
from alpha_worms.reflex_endpoint.game_engine import GameEngine # Replace with your actual game import
from alpha_worms.mcts import MCTS       # Replace with your actual MCTS import
from alpha_worms.nn import WormsPredictor
import time
import numpy as np
 # Replace with your actual Model import
def play_match(game, p1_nnet, p2_nnet, args, verbose=True):
    """Plays a single game between two neural networks."""
    state = game.get_initial_state()
    
    # Each player gets their own isolated MCTS tree
    mcts_p1 = MCTS(game, p1_nnet, args)
    mcts_p2 = MCTS(game, p2_nnet, args)
    
    move_count = 1

    while True:
        current_player = state.board_map.current_player
        
        if verbose:
            print(f"\n{'='*30}")
            print(f"Turn {move_count} | Player {current_player}'s Action")
            print(f"{'='*30}")
            
            # 💡 NOTE: If you haven't written a display() function yet, 
            # you can just print(state.board_map) here for now.
            try:
                game.display(state) 
            except AttributeError:
                print(state.board_map) 

        # Decide whose brain to use
        active_mcts = mcts_p1 if current_player == 1 else mcts_p2
        
        # Canonicalize the state
        canonical_state = game.get_canonical_form(state)
        
        # Get action probabilities (temp=0 means absolute best move only)
        pi = active_mcts.get_action_probs(canonical_state, temp=0)
        action = np.argmax(pi)
        
        if verbose:
            print(f"\n-> Player {current_player} selected Action: {action}")
            time.sleep(0.5)  # Pause for half a second so you can read it!
            
        # Apply the move
        state = game.get_next_state(state, action)
        move_count += 1
        
        # Check for game over
        winner = game.get_winner(state)
        if winner != 0:
            if verbose:
                print(f"\n{'='*30}")
                print(f"🏆 GAME OVER 🏆")
                try:
                    game.display(state)
                except AttributeError:
                    print(state.board_map)
                
                # Winner usually returns 1 (P1), -1 (P2), or something small like 0.1 for a draw
                result = "Player 1" if winner == 1 else "Player 2" if winner == -1 else "Draw"
                print(f"Result: {result} wins after {move_count} turns!")
                
            return winner

def evaluate_checkpoints(games_to_play=20):
    print("⚔️ Welcome to the Arena ⚔️")
    game = GameEngine()
    
    # We can use a tiny args dict for evaluation (fewer sims needed since temp=0)
    eval_args = {'numMCTSSims': 50, 'cpuct': 1.0}

    # 1. Load the Challenger (The latest checkpoint)
    print("Loading Checkpoint 25 (The Master)...")
    nnet_latest = WormsPredictor()
    nnet_latest.nnet.load_checkpoint(folder='./checkpoints', filename='checkpoint_25.pth.tar')
    
    # 2. Load the Baseline (The untrained/early checkpoint)
    print("Loading Checkpoint 1 (The Novice)...")
    nnet_baseline = WormsPredictor()
    nnet_baseline.nnet.load_checkpoint(folder='./checkpoints', filename='checkpoint_1.pth.tar')

    latest_wins = 0
    baseline_wins = 0
    draws = 0

    print(f"\nPlaying {games_to_play} games...")
    
    for i in range(1, games_to_play + 1):
        # 💡 Set to True for the first game so you can watch it visually.
        # Set to False for the rest so it quickly simulates the remaining matches.
        watch_game = (i == 1) 
        
        # Swap who goes first to make it fair
        if i % 2 != 0:
            if watch_game:
                print(f"\n{'*'*40}\nGame {i}: Master (P1) vs Novice (P2)\n{'*'*40}")
            else:
                print(f"Game {i}: Master (P1) vs Novice (P2)...", end=" ")
                
            # Play the match!
            winner = play_match(game, nnet_latest, nnet_baseline, eval_args, verbose=watch_game)
            
            if winner == 1:
                latest_wins += 1
                print("Master Wins!" if not watch_game else "\n>>> Result: Master (P1) Wins! <<<")
            elif winner == -1:
                baseline_wins += 1
                print("Novice Wins!" if not watch_game else "\n>>> Result: Novice (P2) Wins! <<<")
            else:
                draws += 1
                print("Draw!" if not watch_game else "\n>>> Result: It's a Draw! <<<")
                
        else:
            if watch_game:
                print(f"\n{'*'*40}\nGame {i}: Novice (P1) vs Master (P2)\n{'*'*40}")
            else:
                print(f"Game {i}: Novice (P1) vs Master (P2)...", end=" ")
                
            # Play the match!
            winner = play_match(game, nnet_baseline, nnet_latest, eval_args, verbose=watch_game)
            
            # Since Novice is P1, a winner of 1 means Novice won
            if winner == 1:
                baseline_wins += 1
                print("Novice Wins!" if not watch_game else "\n>>> Result: Novice (P1) Wins! <<<")
            elif winner == -1:
                latest_wins += 1
                print("Master Wins!" if not watch_game else "\n>>> Result: Master (P2) Wins! <<<")
            else:
                draws += 1
                print("Draw!" if not watch_game else "\n>>> Result: It's a Draw! <<<")

    print("\n🏆 Final Results 🏆")
    print(f"Master (Cp 25) Wins: {latest_wins}")
    print(f"Novice (Cp 1) Wins : {baseline_wins}")
    print(f"Draws              : {draws}")

if __name__ == "__main__":
    evaluate_checkpoints(20)