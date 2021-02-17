import gym
import random
import requests
import numpy as np
from Lib import copy
from gym_connect_four import ConnectFourEnv
from datetime import datetime

env: ConnectFourEnv = gym.make("ConnectFour-v0") #The game environment

def player_move(env):
   env.change_player() #Change to player

   avmoves = env.available_moves()
   if not avmoves:
      env.change_player() #Change back to agent before returning
      return -1

   action = -1
   if autoplay == 1:
      action = random.choice(list(avmoves))
   else:
      while action > 6 or action < 0:
         print("Make your move (1-7):")
         inpt = input()
         try: 
            inpt = int(inpt)
            action = inpt - 1
         except ValueError:
            print("Please enter a number!")

   state, reward, done, _ = env.step(action)

   if done:
      if reward == 1: #Reward is always in current player's view
         reward = -1
   env.change_player() #Change back to agent before returning
   return state, reward, done

def agent_move():
   if (print_debug): move_start_time = datetime.now() #Save time before calculating the move
   
   move = alpha_beta_decision() #Decide the best move

   if (print_debug): #Calculate move time
      move_end_time = datetime.now() 
      total_move_time = move_end_time - move_start_time
      print("Total move time: " + str(total_move_time))

   return move

def alpha_beta_decision():
   print("Thinking...")
   if (print_debug):
      print("Using depth " + str(depth))
   
   start_depth = depth
   val = -np.inf
   move = -10
   values = [None] * 7
   debug = []

   actions = env.available_moves()

   for a in actions: #For each legal move, make it
      succ_board = copy.deepcopy(env)
      succ_state, result, _, _ = succ_board.step(a)

      if result == 1: #If win in one move is possible, play that move
         print("I have made the move " + str(a + 1))
         if (print_debug):
            print("which is a winning move")
         return a

      new_val = min_value(succ_board, -np.inf, np.inf, start_depth) #Beginning of recursion
      
      debug.append((a, new_val, succ_state)) #Saving for debugging

      if new_val >= val: #Find the best move
         val = new_val
         move = a
      
      values[a] = new_val #Saving to print, for debugging
      
   print("I have made the move " + str(move + 1))
   
   if (print_debug): #Debug printout
      print("which has the value: " + str(val))
      print("Values: " + str(values))
   return move

def max_value(board, alpha, beta, depth):
   board.change_player()

   if depth <= 0: #Check if depth is reached
      return evaluate(board.board, 0)

   v = -np.inf
   actions = board.available_moves()
   
   for a in actions: #For each legal move, make it
      succ_board = copy.deepcopy(board)
      succ_state, result, done, _ = succ_board.step(a)
      
      #TODO: This loop will break even if a loss or a draw is found, but there might be a win further to
      #the right. FIX!
      if not result == 0 and done: #If some result was reached, return the utility of this board
         return evaluate(succ_state, result)
      else: #If no result was reached, go one level deeper
         v = max(v, min_value(succ_board, alpha, beta, depth - 1)) #Recursive call
         if v >= beta: #If the maximum value is larger than beta, return it
            return v
         alpha = max(alpha, v) #Update alpha

   return v

def min_value(board, alpha, beta, depth):
   board.change_player()

   if depth <= 0: #Check for depth
      return evaluate(board.board, 0)

   v = np.inf
   actions = board.available_moves()

   for a in actions: #For each legal move, make it
      succ_board = copy.deepcopy(board)
      succ_state, result, done, _ = succ_board.step(a)

      if not result == 0 and done: #If some result was reached, return the utility of this board
         if result == 1: #If win, invert it, since we are "playing as the opponent"
            result = -1

         return evaluate(succ_state, result) 
      else: #If no result was reached, go one level deeper
         v = min(v, max_value(succ_board, alpha, beta, depth - 1))
         if v <= alpha:
            return v
         beta = min(beta, v)
      
   return v

def evaluate(state, result):
   if result == -1: #If loss, return negative infinity
      return -np.inf
   elif result == 0.5: #If draw, return 0
      return 0
   elif result == 1: #If win, return infinity
      return np.inf
   else: #If not terminal state, return an estimated utility value
      winning_lines_reference =[
      [3, 4, 5, 7, 5, 4, 3],
      [4, 6, 8, 10, 8, 6, 4],
      [5, 8, 11, 13, 11, 8, 5],
      [5, 8, 11, 13, 11, 8, 5],
      [4, 6, 8, 10, 8, 6, 4],
      [3, 4, 5, 7, 5, 4, 3]]

      value = 0
      
      for i in range(0, 5):
         for j in range(0,6):
            value += winning_lines_reference[i][j] * state[i][j]

      return value #random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])

def play_game():
   """
   The reward for a game is as follows. You get a
   botaction = random.choice(list(avmoves)) reward from the
   server after each move, but it is 0 while the game is running
   loss = -1
   win = +1
   draw = +0.5
   error = -10 (you get this if you try to play in a full column)
   Currently the player always makes the first move
   """

   agent_gets_move = False
   state = None

   agent_gets_move = initialize_game(agent_gets_move, state)

   play_loop(agent_gets_move, state)

def play_loop(agent_gets_move, state):
   done = False
   while not done: #Make both agent and player moves
      if agent_gets_move: 
         agmove = agent_move() #Select your move
         available_moves = env.available_moves()
         if agmove not in available_moves: #Check if move is legal
            print("Agent tried to make an illegal move! Games ends.")
            break
         state, result, done, _ = env.step(agmove) #Make move

      agent_gets_move = True #Agent only skips move first turn if player starts

      env.render()

      if not done: #Player makes move, returns reward from agent's view
         state, result, done = player_move(env)

      #Check if the game is over
      if result != 0:
         done = True
         print("---------------------------------------------------\n")
         print("GAME OVER \n")
         if result == 1:
            print("You lost :(")
         elif result == 0.5:
            print("It's a draw!")
         elif result == -1:
            print("You won!")
         else:
            print("Unexpected result result={}".format(result))
         print("\nFinal board:")
         env.render()  #Print final gamestate
         print()
         print("Press enter to play again")
         input()
      else:
         print("---------------------------------------------------\n")
         print("Current board (you play as X):")
         env.render() #Print current gamestate
         print()

def initialize_game(agent_gets_move, state):
   env.reset(board=None )#Reset game to starting state

   agent_gets_move = random.choice([True, False]) #Determine first player
   if agent_gets_move:
      print("---------------------------------------------------\n")
      print('Bot starts!\n')
      env.render() #Print current gamestate
      print()
   else:
      print("---------------------------------------------------\n")
      print('You start!\n')
   return agent_gets_move

def set_difficulty():
   global depth
   global print_debug
   global autoplay

   difficulty = -1
   while difficulty < 0 or difficulty > 5:
      print("Select a difficulty between 0 and 5")
      print("(Higher difficulty -> longer move times):")
      try:
         difficulty = int(input())
      except ValueError:
         print("Please enter a number!")
   depth = difficulty
   print_debug = False
   autoplay = 0

def advanced_settings():
   global depth
   global print_debug
   global autoplay

   print("--ADVANCED SETTINGS--")
   print("Do you want to see debug printouts (y/n)?")
   debug_selection = None
   while not (debug_selection == 'y' or debug_selection == 'n'):
      debug_selection = input()
   if debug_selection == 'y':
      print_debug = True
   else:
      print_debug = False

   print("Select autoplay option:")
   print("(0: no autoplay, 1: against random moves)")
   autoplay_choice = -1
   while autoplay_choice > 1 or autoplay_choice < 0:
      try:
         autoplay_choice = int(input())
      except ValueError:
         print("Please enter a number!")
   autoplay = autoplay_choice

   difficulty = -1
   print("Select search depth:")
   while difficulty < 0:
      try:
         difficulty = int(input())
      except ValueError:
         print("Please enter a number!")
   depth = difficulty

def main():
   print("---------------------------------------------------\n \n")
   print("Welcome to Connect Four!\n")
   print("Press enter to continue:")
   print("(Enter 'a' for advanced settings)\n")

   if input() == 'a':
      advanced_settings()
   else:
      set_difficulty()

   print("\nYou play as \'X\'. Press enter to begin!")
   input()

   while True:
      play_game()

if __name__ == "__main__":
   main()