# different understanding of MCTS

# three main trees
# Q -> mapping from states,action pairs to values #       Q is updated while we are traversing the mcts tree
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

class torch_policy_value_model(nn.Module):
    def __init__(self):
        super(torch_policy_value_model, self).__init__()
        self.input_layer = nn.Linear(9,256)
        self.hidden_layer = nn.Linear(256,512)
        self.hidden_layer2 = nn.Linear(512,512)
        self.hidden_layer3 = nn.Linear(512,512)
        self.sigmoid = nn.Sigmoid()
        self.value_hidden = nn.Linear(512,2048)
        self.value_output = nn.Linear(2048,1)
        self.policy_hidden = nn.Linear(512,2048)
        self.policy_output = nn.Linear(2048,9)

    def forward(self,obs):
        x = self.input_layer(obs)
        x = torch.tanh(x)
        x = self.hidden_layer(x)
        x = torch.tanh(x)
        x = self.hidden_layer2(x)
        x = torch.tanh(x)
        x = self.hidden_layer3(x)
        x = torch.tanh(x)

        policy  = self.policy_hidden(x)
        policy  = self.policy_output(policy)
        policy  = self.sigmoid(policy)

        value = self.value_hidden(x)
        value  = self.value_output(value)
        value = torch.tanh(value)

        return policy,value


class model_wrapper():
    # heres what we want our access to be
    # forward -> value,policy
    # train
    #   takes in (state,action,new_state,reward)
    #   internally trains the model
    def __init__(self,policy_value_model=None):
        # create the model
        if(policy_value_model == None):
            self.model = torch_policy_value_model()
        else:
            self.model = policy_value_model
        learning_rate = 0.0001

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.policy_loss = nn.PoissonNLLLoss()
        # self.policy_loss = MSELoss()
        # self.policy_loss = nn.CrossEntropyLoss()
        self.value_loss = MSELoss()
        


    def forward(self,obs):
        obs = torch.Tensor(obs)
        policy,value = self.model(obs)
        return policy,value

    def convert_action_onehot(self,actions):
        onehots = []
        for action in actions:
            onehot = [1 if i == action else 0 for i in range(9)]
            onehots.append(onehot)
        return onehots

    def train(self, data):
        # shuffle data before we do anything with it
        np.random.shuffle(data)



        # data should be in state,action,new_state,reward
        obs,actions,values = zip(*data)

        # reformat data 
        obs = torch.Tensor(obs).float()
        values = torch.Tensor(values).float().view((-1,1))

        # feed these through the network
        # cross entropy loss requires an index so we are taking the best
        # action as our index
        #actions = torch.Tensor(np.argmax(actions,axis=1)).long()
        actions = torch.Tensor(actions).float()


        epochs = 50
        for i in range(epochs):
            # Update the value network
            self.optimizer.zero_grad()

            policy_prediction, value_prediction = self.model(obs)
            v_loss = self.value_loss(value_prediction,values)

            # compute policy loss
            p_loss = self.policy_loss(policy_prediction,actions)

            total_loss = v_loss+p_loss
            total_loss.backward()
            print("\rLoss:"+str(total_loss)+"                  ",end="")

            self.optimizer.step()
            


class tabular_mcts:

    def __init__(self,number_actions=9,policy_value_model=None):
        self.Q = {}
        self.N = {}
        self.P = {}
        self.number_actions = number_actions
        self.tictactoe_functions = tictactoe_methods()
        self.model = model_wrapper(policy_value_model=policy_value_model)
    
    def train(self,experience):
        # pass through function for our modesl
        self.model.train(experience)

    def clear_tree(self):
        # clears everything except what the model has learned
        self.Q = {}
        self.N = {}
        self.P = {}


    def serialize_board(self,board):
        # we know numbers will only be 0,1,2
        # so we are going to join them into a string
        serialized_string = "".join([str(x) for x in board])
        return serialized_string

    def seen(self,board):
        serialized_board = self.serialize_board(board)
        return serialized_board in self.Q.keys()

    def get_Q(self,board):
        serialized_board = self.serialize_board(board)

        return self.Q[serialized_board]

    def get_N(self, board):
        serialized_board = self.serialize_board(board)
        return self.N[serialized_board]

    def get_action_list(self, board,turn):
        # ask the model for the N's of a board
        # we have to flip boards on turn 2
        if(turn == 2):
            board = self.tictactoe_functions.flip_board(board)
        serialized_board = self.serialize_board(board)
        return self.N[serialized_board]

    def get_P(self,board):
        serialized_board = self.serialize_board(board)

        return self.P[serialized_board]


    def update_Q(self,board,action,value):
        # update the moving average for all rotated boards
        serialized_board = self.serialize_board(board)
        N_state_action = self.get_N(board)[action]
        Q_state_action = self.get_Q(board)[action]
        # update the moving average

        self.Q[serialized_board][action]  = (N_state_action*Q_state_action + value) / (N_state_action + 1)

    def increment_N(self,board,action):

        serialized_board = self.serialize_board(board)
        self.N[serialized_board][action]  += 1

    def set_P(self,board,policy):
        serialized_board = self.serialize_board(board)
        self.P[serialized_board] = policy

    def call_model(self,board):
        #value = random.random() - .5
        policy,value = self.model.forward(board)
        return [policy, value]

    def check_for_winner(self,board):
        winner = self.tictactoe_functions.get_winner(board)
        # -1 is no one has won yet
        if(winner == -1):
            return -2
        # if its 0 its a tie
        if(winner == 0):
            # tie
            return 0
        # if its 1 player 1
        if(winner ==  1):
            return 1
        # if its 2 player 2 has one, but we want to represent that as -1
        if(winner == 2):
            return -1

    def get_rotated_boards(self,board,turn=1):
        # return all 4 boards that are the same but rotated
        return self.tictactoe_functions.get_rotated_boards(board)


    def expand_node(self,board):
        # first add this to Q,N
        serialized_board =  self.serialize_board(board)
        self.Q[serialized_board] = [0 for i in range(self.number_actions)]
        self.N[serialized_board] = [0 for i in range(self.number_actions)]



    def simulate_step(self,board,turn,debug=False):

        # if the turn is 2 flip the board,
        # that way we are always looking from x perspective
        if(turn == 2):
            board = self.tictactoe_functions.flip_board(board)

        # check if this is a winning board
        winner = self.check_for_winner(board)
        if(winner != -2):
            return -winner

        # check board to see if we have seen this before
        if(not self.seen(board)):
                
            # we haven't seen this board before and we know
            # its not a winning board
            # gotta expand this node
            self.expand_node(board)
            predicted_policy, predicted_value = self.call_model(board)
            predicted_policy = [float(x) for x in predicted_policy]
            self.set_P(board, predicted_policy)
            predicted_value = float(predicted_value)
            return  -predicted_value

        # we have seen this position before, so we have to go deeper

        # to know which node we want to select to explore we use an interesting formula
        # which is U[s][a] = Q[s][a] + c_puct*P[s][a] * sqrt(sum(N[s]))/(1+N[s][a])
        c_puct = 1.0 # used for exploration

        # U stands for Upper confidence bound

        best_U =  -float("inf")
        multiple_best_U = []
        best_action = 0

        U_list = []
        
        N_s = self.get_N(board)
        Q_s = self.get_Q(board)
        P_s = self.get_P(board)
        for action in self.tictactoe_functions.get_possible_actions(board):
            Q_s_a = Q_s[action]
            P_s_a = P_s[action]
            if(turn == 2):
                pass
                # the other users turn
                # inverse the Q,P values
                #Q_s_a = -Q_s_a
                # P_s_a = -P_s_a
            # print(turn, Q_s_a)
            # N_term = math.sqrt(sum(N_s))/(N_s[action]+1)
            N_term = math.sqrt(sum(N_s))/(N_s[action]+1)

            #U = Q_s_a + c_puct*P_s_a*N_term
            U = Q_s_a + c_puct*N_term


            U_list.append((action,U))

            if(U > best_U):
                best_U = U
                best_action = action
        # check if they are all the same 


        if(debug):
            Q_s = self.get_Q(board)
            P_s = self.get_P(board)
            print("--------------")
            print("board:")
            self.tictactoe_functions.pretty_print(board)
            print("U:",U_list)
            print("N:",N_s)
            print("Q:",Q_s)
            print("P:",P_s)
            print("possible",self.tictactoe_functions.get_possible_actions(board))
            print("best one", best_action)

        # unflip the board before an action is taken
        regular_board = board
        if(turn == 2):
            regular_board = self.tictactoe_functions.flip_board(board)

            
        # print(best_action)
        board_after_action = self.tictactoe_functions.get_next_board(regular_board,best_action,turn)

        next_turn = 2 if turn == 1 else 1
        # print("\tgoing down on this board",board,action)
        value_below = self.simulate_step(board_after_action,next_turn,debug=debug)
        # update Q and N
        # updating these values on the flipped board
        self.update_Q(board, best_action, value_below)
        self.increment_N(board, best_action)

        return -value_below

    def copy(self):
        # copy both neural networks

        policy_value_network = torch_policy_value_model()
        policy_value_network.load_state_dict(self.model.model.state_dict())
        copy = tabular_mcts(policy_value_model=policy_value_network)
        return copy



