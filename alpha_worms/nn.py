import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import threading
import queue
import time
from alpha_worms.reflex_endpoint.dto import GameDTO
from alpha_worms.reflex_endpoint.tensor_engine import TensorEngine
from alpha_worms.reflex_endpoint.game_engine import GameEngine

torch.set_num_threads(1) # Important for multiprocessing to prevent thread contention

class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual # Skip connection
        return F.relu(out)

class WormsNet(nn.Module):
    def __init__(self, input_planes=55, board_slots=33, action_size=25):
        super(WormsNet, self).__init__()
        self.num_channels = 128 # The "Brain Power" hyperparameter
        
        # 1. Initial Block: Takes 55 planes and expands them to 128
        self.conv_input = nn.Conv1d(input_planes, self.num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm1d(self.num_channels)
        
        # 2. Residual Tower: 10 layers of deep thinking
        self.res_tower = nn.Sequential(
            *[ResBlock1D(self.num_channels) for _ in range(10)]
        )
        
        # 3. Policy Head: Predicts the best of the 25 moves
        self.policy_conv = nn.Conv1d(self.num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm1d(2)
        self.policy_fc = nn.Linear(2 * board_slots, action_size)
        
        # 4. Value Head: Predicts if the current player is winning (-1 to 1)
        self.value_conv = nn.Conv1d(self.num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm1d(1)
        self.value_fc1 = nn.Linear(board_slots, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, s):
        # s shape: (batch_size, 55, 33)
        s = F.relu(self.bn_input(self.conv_input(s)))
        s = self.res_tower(s)
        
        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(s)))
        p = p.view(p.size(0), -1) # Flatten to (batch, 66)
        p = F.log_softmax(self.policy_fc(p), dim=1)
        
        # Value Head
        v = F.relu(self.value_bn(self.value_conv(s)))
        v = v.view(v.size(0), -1) # Flatten to (batch, 33)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v
    
# --- 2. THE WRAPPER CLASS ---
class WormsPredictor:
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.nnet = WormsNet().to(self.device)
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=0.001, weight_decay=1e-4)
        self.tensor_engine = TensorEngine()
        self.share_memory() # Enable sharing for multiprocessing
        self.load_checkpoint(folder='./checkpoints/', filename='checkpoint_25.pth.tar')
        print(f"Using device: {self.device}")

        # 1. THE QUEUE SYSTEM
        self.prediction_queue = queue.Queue()
        self.batch_size = 32 # Maximum items to scoop up at once
        
        # 2. START THE GPU WORKER THREAD
        self.running = True
        self.worker_thread = threading.Thread(target=self._gpu_worker_loop, daemon=True)
        self.worker_thread.start()

    def share_memory(self):
        """
        Allows the underlying PyTorch model weights to be shared across 
        multiple CPU cores for parallel self-play.
        """
        self.nnet.share_memory()

    def predict(self, state):
        """
        Input: Numpy array of shape (55, 33)
        Output: Policy (numpy), Value (float)
        Used by MCTS.
        """
        state = self.tensor_engine.to_tensor(state)

        # Prepare state for the network
        state_tensor = torch.FloatTensor(state.astype(np.float32)).to(self.device)
        state_tensor = state_tensor.view(1, 55, 33) # Add batch dimension

        self.nnet.eval() # Set to evaluation mode
        with torch.no_grad():
            pi, v = self.nnet(state_tensor)

        # Convert back to numpy for MCTS to use
        # exp() because the model returns log_softmax
        return torch.exp(pi).cpu().numpy()[0], v.cpu().numpy()[0][0]

    def train(self, examples):
        """
        Input: List of (state, pi, v)
        Used by Worms main loop.
        """
        self.nnet.train() # Set to training mode
        
        # Unpack examples
        states, target_pis, target_vs = list(zip(*examples))

        encoded_states = [self.tensor_engine.to_tensor(state) for state in states]
        try:
            new_states = torch.FloatTensor(np.array(encoded_states)).to(self.device)
            target_pis = torch.FloatTensor(np.array(target_pis)).to(self.device)
            target_vs = torch.FloatTensor(np.array(target_vs)).to(self.device)
        except Exception as e:
            print('examples',examples)
            print(f"Error converting training examples to tensors: {e}")
            raise e
        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        out_pi, out_v = self.nnet(new_states)

        # LOSS CALCULATION
        # 1. Policy Loss: Negative Log Likelihood (Cross Entropy)
        # target_pis are the MCTS visit counts
        l_pi = -torch.sum(target_pis * out_pi) / target_pis.size()[0]
        
        # 2. Value Loss: Mean Squared Error
        l_v = torch.sum((target_vs - out_v.view(-1)) ** 2) / target_vs.size()[0]

        total_loss = l_pi + l_v
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()


    def save_checkpoint(self, folder='checkpoints', filename='checkpoint.pth.tar'):
        """
        Saves the current weights and optimizer state to a file.
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        # We save a dictionary containing the model weights AND the optimizer state
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_checkpoint(self, folder='checkpoints', filename='checkpoint.pth.tar'):
        """
        Loads weights into the NN from a saved file.
        """
        # Get absolute path and include it before the folder
        abspath = '/home/pablo/Documents/projects/alpha_worms/'
        filepath = os.path.join(abspath, folder, filename)
        if not os.path.exists(filepath):
            print(f"No checkpoint found at {filepath}. Starting with fresh weights.")
            return

        # Load the file from disk
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Here are the built-in PyTorch methods being used:
        self.nnet.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Model loaded from {filepath}")
    
    def nn_next_move(self, game_dto: GameDTO):
        # Placeholder for neural network inference logic
        # self.load_checkpoint(folder='./checkpoints/', filename='checkpoint_25.pth.tar')
        # Here you would typically pass the tensor through your trained model to get move probabilities
        # For demonstration, we'll just return a dummy move
        game = GameEngine()
        # 3. Get a starting board
        initial_state = game_dto

        # 4. Ask the brain what it thinks! (Get RAW probabilities)
        raw_policy, value = self.predict(initial_state)

        # ---------------------------------------------------------
        # 5. MASK AND RE-NORMALIZE (Same logic as MCTS Expansion)
        # ---------------------------------------------------------
        valid_moves = game.get_allowed_moves(initial_state)

        # Multiply raw probabilities by the valid moves mask (illegal moves become 0.0)
        masked_policy = raw_policy * valid_moves
        sum_probs = np.sum(masked_policy)

        # Re-normalize so the remaining valid probabilities sum to 1.0
        if sum_probs > 0:
            policy = masked_policy / sum_probs
        else:
            # Fallback in case the untrained network output 0.0 for all valid moves
            policy = valid_moves / np.sum(valid_moves)
        # ---------------------------------------------------------

        # Grab the indices of the highest probabilities from the MASKED policy
        top_moves = np.argsort(policy)[::-1][:3]

        return game.get_next_state(initial_state, top_moves[0])

    ## GPU methods:
    def request_prediction(self, state):
        """
        Game threads call this INSTEAD of predict().
        It drops the state in the bucket and waits.
        """
        # Create a personal event lock and a dictionary to hold the answer
        event = threading.Event()
        result = {}
        
        # Put our request in the queue
        self.prediction_queue.put((state, event, result))
        
        # Put this specific game thread to sleep until the GPU unlocks it
        event.wait() 
        
        # Wake up and return the results!
        return result['pi'], result['v']

    def _gpu_worker_loop(self):
        """
        This runs infinitely in the background, scooping up requests.
        """
        while self.running:
            requests = []
            
            # Scoop items out of the queue until it's empty OR we hit our batch limit
            while not self.prediction_queue.empty() and len(requests) < self.batch_size:
                try:
                    requests.append(self.prediction_queue.get_nowait())
                except queue.Empty:
                    break
            
            if len(requests) == 0:
                # If the bucket is empty, sleep for 1 millisecond so we don't fry the CPU
                time.sleep(0.001)
                continue
                
            # --- WE HAVE A BATCH! DO THE MATH! ---
            
            # 1. Unpack the requests
            states = [req[0] for req in requests]
            events = [req[1] for req in requests]
            results = [req[2] for req in requests]
            
            # 2. Convert ALL states into one massive tensor block
            encoded_states = [self.tensor_engine.to_tensor(s) for s in states]
            
            # ADD .astype(np.float32) HERE
            numpy_batch = np.array(encoded_states).astype(np.float32) 
            batch_tensor = torch.FloatTensor(numpy_batch).to(self.device)
            
            # 3. ONE Neural Network Pass
            self.nnet.eval()
            with torch.no_grad():
                pi_batch, v_batch = self.nnet(batch_tensor)
                
            pi_batch = torch.exp(pi_batch).cpu().numpy()
            v_batch = v_batch.cpu().numpy()
            
            # 4. Hand the answers back and wake up the game threads!
            for i in range(len(requests)):
                results[i]['pi'] = pi_batch[i]
                results[i]['v'] = v_batch[i][0]
                events[i].set() # WAKE UP!