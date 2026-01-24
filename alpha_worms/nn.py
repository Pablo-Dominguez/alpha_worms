import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from alpha_worms.reflex_endpoint.tensor_engine import TensorEngine

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
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        target_pis = torch.FloatTensor(np.array(target_pis)).to(self.device)
        target_vs = torch.FloatTensor(np.array(target_vs)).to(self.device)

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        out_pi, out_v = self.nnet(states)

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
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            print(f"No checkpoint found at {filepath}. Starting with fresh weights.")
            return

        # Load the file from disk
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Here are the built-in PyTorch methods being used:
        self.nnet.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Model loaded from {filepath}")

    