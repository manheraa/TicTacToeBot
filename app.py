from flask import Flask, render_template, request, jsonify
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)

# Define your GCN model
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

# Initialize model and load state
model = GCN(input_dim=1, hidden_dim=16, output_dim=3)
model.load_state_dict(torch.load('path_to_saved_model.pth'))
model.eval()

# Define adjacency matrix for a 3x3 grid (8-connected neighborhood)
edge_index = torch.tensor([
    [0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8],  # Horizontal
    [0, 3], [3, 6], [1, 4], [4, 7], [2, 5], [5, 8],  # Vertical
    [0, 4], [4, 8], [2, 4], [4, 6]                   # Diagonals
], dtype=torch.long).t().contiguous()

def prepare_board_state(moves):
    """Prepare the board state from moves, handling '?'. """
    board = [0] * 9
    for i, move in enumerate(moves):
        if move == '?':
            board[i] = 0  # Set '?' positions to 0 (indicating empty or not yet filled)
        else:
            move = int(move)
            board[move] = 1 if i % 2 == 0 else -1  # Player 1's move is 1, Player 2's move is -1
    features = torch.tensor(board, dtype=torch.float).view(-1, 1)
    return features
def predict_next_moves(board):
    """Predict the next moves for the AI based on the current board state."""
    possible_moves = [i for i, v in enumerate(board) if v == 0]  # Find empty spaces
    predictions = []

    for move in possible_moves:
        # Simulate the board state with the new move
        new_board = board.copy()
        new_board[move] = 1  # Assuming it's Player 1's turn (1)

        # Prepare features and data for the model
        features = torch.tensor(new_board, dtype=torch.float).view(-1, 1)
        data = Data(x=features, edge_index=edge_index)

        # Perform inference
        with torch.no_grad():
            output = model(data.x, data.edge_index, torch.tensor([0]))
            probabilities = F.softmax(output, dim=1).squeeze()  # Apply softmax to get probabilities
            
            # Debug: Print output and probabilities
            print(f'Board State for Move {move}: {new_board}')
            print(f'Model Output: {output}')
            print(f'Probabilities: {probabilities}')
            
            if probabilities.numel() == 0:
                print(f"No predictions available for move {move}.")
                continue
            
            predicted_class = torch.argmax(probabilities).item()  # Get the index of the max probability
            predictions.append((move, probabilities.tolist()))
    
    # Sort predictions by the probability of the winning move (index 1)
    predictions.sort(key=lambda x: x[1][1], reverse=True)
    
    return predictions
def check_win(board):
    """Check if there is a win condition on the board."""
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] != 0:
            return board[condition[0]]  # Return 1 or -1 for winning player
    return None

def find_blocking_move(board):
    """Find a move that blocks the opponent from winning."""
    possible_moves = [i for i, v in enumerate(board) if v == 0]  # Find empty spaces
    for move in possible_moves:
        test_board = board.copy()
        test_board[move] = -1  # Test as if the opponent played this move
        if check_win(test_board) == -1:  # Check if this move would result in a win for the opponent
            return move
    return None

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/play', methods=['POST'])
def play():
    data = request.json
    board = data['board']
    user_move = data['user_move']
    
    # Update the board with the user's move
    board[user_move] = -1  # User is Player 2
    
    # Check if the user won
    if check_win(board) is not None:
        return jsonify({'status': 'win', 'board': board})
    
    # Check for a draw
    if all(cell != 0 for cell in board):
        return jsonify({'status': 'draw', 'board': board})
    
    # AI's turn (predict the next move)
    # No need to convert the board to 'moves' format, directly pass the board
    block_move = find_blocking_move(board)
    if block_move is not None:
        best_move = block_move
    else:
        next_moves_predictions = predict_next_moves(board)
        
        # Choose the best move with the highest probability
        best_move = None
        for move, probs in next_moves_predictions:
            if board[move] == 0:  # Ensure the move is still possible
                best_move = move
                break
        
        if best_move is None:
            # If no move is possible (shouldn't happen in a valid game state), just pick the first available move
            best_move = next((i for i, cell in enumerate(board) if cell == 0), None)
    
    if best_move is not None:
        board[best_move] = 1  # AI is Player 1
    
    # Check if the AI won
    if check_win(board) is not None:
        return jsonify({'status': 'ai_win', 'board': board})
    
    # Check for a draw after AI's move
    if all(cell != 0 for cell in board):
        return jsonify({'status': 'draw', 'board': board})
    
    # Continue the game
    return jsonify({'status': 'continue', 'board': board})

