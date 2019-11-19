# different understanding of MCTS

# three main trees
# Q -> mapping from states,action pairs to values
#       Q is updated while we are traversing the mcts tree
# P -> produced from the neural network and it is a policy from a state
# N -> maps states to the amount of times we have visited them

# recap
# Q[start_state][action] -> value of taking action
# P[start_state][action] -> neural networks take on value
# N[start_state][action] -> number of times we have taken this path


# web.stanford.edu/~surag/posts/alphazero.html
from tictactoe_module import tictactoe_methods
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.loss import MSELoss, CrossEntropyLoss
from torch.nn.modules.activation import ReLU, Tanh
import torch.optim as optim
import sys
from os import path
from tabular_model import tabular_mcts,torch_policy_value_model


def run_game_with_human(mcts_model,human_player_pos):

    tictactoe_functions = tictactoe_methods()
    board = tictactoe_functions.get_initial_board()
    turn = 1
    for game_steps in range(11):
        print("Turn #%d:"%turn)
        tictactoe_functions.pretty_print(board)
        if(turn % 2 == human_player_pos):
            # robot turn
            # simulating the games
            simulation_steps = 300
            for i in range(simulation_steps):
                #print('sim:',i)
                mcts_model.simulate_step(board,turn)

            # getting the actions from the mcts tree
            searched_board = board
            if(turn == 2):
                searched_board = tictactoe_functions.flip_board(board)
            actions_list = [n for n in mcts_model.get_N(searched_board)]
        
            action = np.argmax(actions_list)
            print("Visit counts:",actions_list)
            print("value of each next state:",mcts_model.get_Q(searched_board))
            print("policy:",mcts_model.get_P(searched_board))
        else:
            # player turn
            x,y = [int(x) for x in input("input[0-2] x y: ").split()]
            action = y*3 + x

        board = tictactoe_functions.get_next_board(board, action, turn)

       

        winner = tictactoe_functions.get_winner(board)
        if(winner != -1):
            tictactoe_functions.pretty_print(board)
            return winner
        turn = 2 if turn == 1 else 1



if __name__ == "__main__":
    # create this model and 
    policy_value_model = torch.load("policy_value_model.torch")

    human_player_position = int(input("Do you want to go first[0] for second[1]: "))
    # inialize the model
    mcts_model = tabular_mcts(policy_value_model=policy_value_model)

    winner = run_game_with_human(mcts_model,human_player_pos=human_player_position)
    print('-----------')
    if(winner == 0):
        print("Tie!")
    elif(winner == (human_player_position-1)%2):
        print("Human Player was victorious")
    else:
        print("You were defeated by an AI")

