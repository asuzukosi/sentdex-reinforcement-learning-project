import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
import time
from matplotlib import style

style.use("ggplot")

SIZE = 10 # GRID SIZE
HM_EPISODES = 25_000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25

epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 3000

start_q_table = None # or a file name which we can load in

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1
FOOD_N = 2

ENEMY_N = 3

d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255)}

class Blob:
    # blob object
    def __init__(self):
        # random start position
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)
        
    # print the blob location
    def __str__(self):
        return f"x: {self.x}, y: {self.y}"
    
    # used for calculating distance
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)
    
    def action(self, choice):
        # only diagonal movement
        if choice == 0:
            self.move(1, 1)
        elif choice == 1:
            self.move(-1, -1)
        elif choice == 2:
            self.move(-1, 1)
        elif choice == 3:
            self.move(1, -1)
            

    def move(self, x=False, y=False):
        new_x, new_y = self.x, self.y
        if not x:
            new_x += np.random.randint(-1, 2)
        else:
            new_x += x
            
        if not y:
            new_y += np.random.randint(-1, 2)
        else:
            new_y += y
        
        # if the x and y values are out of 
        # bound keep the values the same
        if new_x < 0 or new_x >= SIZE -1:
            new_x = self.x
        if new_y < 0 or new_y >= SIZE -1:
            new_y = self.y
        
        # update the x and y values
        self.x, self.y = new_x, new_y
        

# generate q table
if start_q_table is None:
    q_table = dict()
    for x1 in range (-SIZE+1, SIZE):
        for y1 in range (-SIZE+1, SIZE):
            for x2 in range (-SIZE+1, SIZE):
                for y2 in range (-SIZE+1, SIZE):
                    # initialize random value for the state
                    q_table[((x1, y1), (x2, y2))] = list(np.random.uniform(-5, 0, 4))


else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)
        
episode_rewards = []      
for episode in range(HM_EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()
    
    
    if episode % SHOW_EVERY == 0:
        print(f"on # {episode}, epsilon: {epsilon}")
        print(f"{SHOW_EVERY} ep meal {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False
        
    episode_reward = 0
    for i in range(200):
        obs = (player - food, player - enemy)
        if np.random.random() > epsilon:
            # choose best action based on argmax values
            action = np.argmax(q_table[obs])
        else:
            # choose random action
            action = np.random.randint(0, 4)
            
        # perform action
        player.action(action)
        
        # TODO: let enemy and food to move
        enemy.move() # agent performs better with enemy and food moving
        food.move()
        
        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = - MOVE_PENALTY
        
        new_obs  = (player - food, player - enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]
        
        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE *(reward + DISCOUNT * max_future_q)
        
        q_table[obs][action] = new_q
        
        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[food.x, food.y] = d[FOOD_N]
            env[enemy.x, enemy.y] = d[ENEMY_N]
            env[player.x, player.y] = d[PLAYER_N]
            
            img = Image.fromarray(env, "RGB")
            img = img.resize((300, 300))
            cv2.imshow("image", np.array(img))
            
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break
        
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY
    
    
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}ma")
plt.xlabel(f"episode #")
plt.show()

# save q table into file
with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)