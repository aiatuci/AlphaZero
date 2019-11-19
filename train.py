# different understanding of MCTS
# three main trees # Q -> mapping from states,action pairs to values #       Q is updated while we are traversing the mcts tree # P -> produced from the neural network and it is a policy from a state # N -> maps states to the amount of times we have visited them

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
import signal
from tqdm import tqdm
import sys



class fixed_size_list():
    def __init__(self,size):
        # keep same size
        # but able to add infinitely
        self.size = size
        self.buffer = []
        self.at_capacity = False
        self.index = 0

    def extend(self,items):
        if(self.at_capacity):
            for item in items:
                self.buffer[self.index] = item
                self.index = (self.index + 1 ) % self.size
        else:
            # add until we are at capacity
            if(len(items) + self.index < self.size):
                self.buffer.extend(items)
            else:
                # split it into two parts
                spliting_point  = self.size - self.index
                extend_data = items[:splitting_point]
                rotating_buffer_data = items[splitting_point:]

                self.extend(extend_data)

                self.at_capacity = True
                
                self.extend(rotating_buffer_data)
    def get_list(self):
        return self.buffer

def get_model_action(model,board,turn):
    # simulating the games
    simulation_steps = 50
    for i in range(simulation_steps):
        #print('sim:',i)
        model.simulate_step(board,turn)

    # getting the actions from the mcts tree
    actions_list = [n for n in model.get_action_list(board,turn)]
    

    # take the action that was visited the most
    return actions_list


def pit(old_model,new_model,number_of_games=25):
    # test the models against each other and accpet the new model
    # if it wins 55% of the non tie games games
    tictactoe_functions = tictactoe_methods()
    winners = {0:0,1:0,2:0}
    # winners = {'tie':0,'old_model':0,'new_model':0}
    turn = 1
    # check the two models are different
    for game in range(number_of_games):
        board = tictactoe_functions.get_initial_board()
        if(game < number_of_games //2):
            # old_model goes first
            player1 = old_model
            player2 = new_model

        else:
            # new model goes first
            player1 = new_model 
            player2 = old_model

        for player_turn in range(9):

            if(player_turn % 2 == 0):
                # get player 1's action
                action_list = get_model_action(player1,board,turn)
                action = np.argmax(action_list) 
            else:
                action_list = get_model_action(player2,board,turn)
                action = np.argmax(action_list) 
                
            if(player_turn == 0):
                # first move completely random
                action = np.random.choice(9,1)[0]



            board = tictactoe_functions.get_next_board(board, action, turn)
            winner = tictactoe_functions.get_winner(board)
            if(winner != -1):
                # update winners array
                if(game >= number_of_games // 2):
                    # flip the winner so it corresponds with the correct model
                    if(winner == 2):
                        winner = 1
                    elif(winner == 1):
                        winner = 2

                winners[winner] += 1
                # clear both players mcts tree
                player1.clear_tree()
                player2.clear_tree()
                break
            turn = 2 if turn == 1 else 1
    return winners



def get_game_action_probs(mcts_model, board, turn, temp=1.0):
    visit_counts = get_model_action(mcts_model,board,turn)
    # number of times each node was visisted under this node

    if(temp == 0):
        # we are going to take the one with
        # the highest visit count
        action_index  = np.argmax(visit_counts)
        # construct the probabilities to train on
        action_probs = [0 for i in range(9)] # don't use [0]*9 python is weird there
        action_probs[action_index]  = 1.0

        return action_probs

    # otherwise we are using temp which 
    action_probs = [x**(1./temp) for x in visit_counts]
    # regularize action_probs 
    action_sum = sum(action_probs)
    action_probs = [x/action_sum for x in action_probs]
    return action_probs


def run_game(mcts_model):

    tictactoe_functions = tictactoe_methods()
    board = tictactoe_functions.get_initial_board()
    experience = []
    turn = 1
    for game_step in range(10):

        # temp controls how likely 
        # the tree will explore
        # 1 for exploration
        # 0 for the pit, always take the best action
        temp = 1.0
        action_probs = get_game_action_probs(mcts_model,board,turn,temp=temp)

        #chose a action from these probabilities
        action = np.random.choice(9,1,p=action_probs)[0]

        if(game_step == 0):
            # truly random for first step
            action = np.random.choice(9,1)[0]


        # initially use a placeholder value for value
        # this experience is [obs, action, value]
        
        # quadruple our experience by rotating the board
        # if turn == 2 we have to flip the board for training
        training_board = board
        if(turn == 2):
            training_board = tictactoe_functions.flip_board(board)
        #experience.append([training_board,action_probs,-9999])

        # rotating boards to get more info
        rotated_boards = tictactoe_functions.get_rotated_boards(training_board)
        rotated_action_probs = tictactoe_functions.get_rotated_boards(action_probs)
        for i in range(4):
            experience.append([rotated_boards[i],rotated_action_probs[i],-9999])


        board = tictactoe_functions.get_next_board(board, action, turn)


        winner = tictactoe_functions.get_winner(board)
        if(winner != -1):
            return winner,experience
            break
        turn = 2 if turn == 1 else 1




def update_experience_value(winner,experience):
    # in place update experience
    winner_value = -8888
    if(winner == 1):
        # player 1 wins
        winner_value = 1 
    elif(winner == 2): 
        winner_value = -1
    elif(winner == 0):
        # tie
        winner_value = 0
    else:
        print("should get here for update experience_value")

    # doing in place to avoid gross amounts of extra memory

    # we have to ossolate winners because we flipped 
    # half the boards
    for index in range(len(experience)):
        experience[index][2] = winner_value
        # will be winner_value when turn == 1
        # and -winner_value when turn == 2
        if(index != 0 and (index) % 4 == 0):
            # because we rotated 4 times
            winner_value = -winner_value







if __name__ == "__main__":

    policy_value_model = None
    if(path.exists("policy_value_model.torch")):
        print("Loading Prexisting model")
        policy_value_model = torch.load("policy_value_model.torch")

    # inialize the model
    mcts_model = tabular_mcts(policy_value_model = policy_value_model)

    # num games per training loop
    num_training_loops = 500
    base_num_games = 100
    num_games = base_num_games

    wins = {0:0,1:0,2:0}



    total_experience = []#fixed_size_list(100000)
    for train_loop in range(num_training_loops):
        for game in tqdm(range(num_games),leave=False):
            winner, experience= run_game(mcts_model)
            
            # update the experience 
            # based on the real winner of the game
            update_experience_value(winner,experience)
            # check to make sure this is updated
            total_experience.extend(experience)
            wins[winner] += 1

            # what would happen if you cleared the tree here?
            #mcts_model.clear_tree()

        #after we have played our games, update the model
        old_model = mcts_model.copy()
        print("\rTraining...",end="")
        mcts_model.train(total_experience)
        total_games =  num_games
        win_averages = [x/total_games for x in wins.values()]
        print("\rTies: %.2f Player 1: %.2f Player 2: %.2f                  "%(win_averages[0],win_averages[1],win_averages[2]))


        # with this set of weights
        wins = {0:0,1:0,2:0}
        # clear out the tree that we have built up 
        mcts_model.clear_tree()

        # run through the pit
        print("\rCompeting in the Pit...",end="")
        num_pit_games = 50
        pit_results = pit(old_model,mcts_model,number_of_games=num_pit_games)
        print("\rpit results:             ")
        print("Ties:",pit_results[0])
        print("Old Model:",pit_results[1])
        print("New Model:",pit_results[2])

        # check if the new model won more than .51 % of the no tie games
        if(pit_results[2] < (num_pit_games-pit_results[0]) * .51):
            # old_model won
            #mcts_model = tabular_mcts()
            mcts_model = old_model
            # go to before training
            num_games *= 2
            total_experience = []
            print("old model won, training for twice as long")

        else:
            print("keeping new model")
            num_games = base_num_games
            num_update_steps = 1
            # we are learning a new policy,
            # remove the old data
            total_experience = []
            print("saving model...")
            torch.save(mcts_model.model.model,"policy_value_model.torch")

        if(num_games >= 64*100):
            print("thats alot of games to play, we are just going to try to play less")
            num_games = base_num_games




        print("---------")


    # save value network
   
    torch.save(mcts_model.model.model,"policy_value_model.torch")
