from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from collections import deque
import numpy as np
import random
import cv2
from PIL import Image
import os
from tqdm import tqdm

from qlearning_env import Blob

OBSERVATION_SPACE_SIZE = 256
ACTION_SPACE_SIZE = 10

REPLAY_MEMORY_SIZE = 50_000
MINIBATCH_SIZE  = 64
MIN_REPLAY_MEMORY_SIZE = 1_000
DISCOUNT = 0.90
UPDATE_TARGET_EVERY = 5
MIN_REWARD = -200
MEMORY_FRACTION = 0.20
EPISODES = 20_000

# epsilon stuff
epsilon = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 50
SHOW_PREVIEW = False



class BlobEnv:
    SIZE = 10
    RETURN_IMAGES = True
    MOVE_PENALTY = 1
    ENEMY_PENALTY = 300
    FOOD_REWARD = 25
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)
    ACTION_SPACE_SIZE = 9
    PLAYER_N = 1
    FODD_N = 2
    ENEMY_N = 3
    
    d = {
        1: (255, 175, 0),
        2: (0, 255, 0),
        3: (0, 0, 255)
    }
    def __init__(self):
        pass
    
    def reset(self):
        self.player = Blob(self.SIZE)
        self.food = Blob(self.SIZE)
        
        # ensure food does not spawn in the same location with the user
        while self.food == self.player:
            self.food = Blob(self.SIZE)
        
        
        self.enemy = Blob(self.SIZE)
        # ensure enemy does not spawn in the same location with the user or the food
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.SIZE)
            
        self.episode_step = 0
        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            observation = (self.player - self.food, self.player - self.enemy)
        return observation
        
    
    def step(self, action):
        self.episode_step += 1
        self.player.action(action)
        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player - self.food, self.player - self.enemy)
        
        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.food:
            reward = self.FOOD_REWARD
        else:
            reward = -self.MOVE_PENALTY
        
        done = False
        if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.episode_step >= 200:
            done = True
            
        return new_observation, reward, done
    
    def render(self):
        img = self.get_image()
        img = img.resize((300, 300))
        cv2.imshow("image", np.array(img))
        cv2.waitKey(1)
        
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        env[self.food.x, self.food.y] = self.d[self.FODD_N]
        env[self.enemy.x, self.enemy.y] = self.d[self.ENEMY_N]
        env[self.player.x, self.player.y] = self.d[self.PLAYER_N]
        img = Image.fromarray(env, "RGB")
        return img
        
    
    
env = BlobEnv()
ep_reward = [-200]

# so results can be same
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

# create models directory if it does not currently exist
if not os.path.isdir("models"):
    os.mkdir("models")


# DQN AGENT
class DQNAgent:
    def __init__(self):
        # create main model loss and optimizer, this is what we train with every step
        self.model, self.loss, self.optimizer = self.create_model()
        # create target model - this is waht we predict with every step
        self.target_model, self.target_loss, self.target_optimizer = self.create_model()
        # copy weights from the main model to the target model
        self.target_model.load_state_dict(self.model.state_dict())
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0
        
        
        
    def create_model(self):
        model: nn.Sequential = nn.Sequential(
            # stack one
            nn.Conv2d(OBSERVATION_SPACE_SIZE, 256, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.2),
            
            # stack two
            nn.LazyConv2d(256, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.2),
            
            # output stack
            nn.Flatten(),
            nn.LazyLinear(64),
            nn.LazyLinear(ACTION_SPACE_SIZE)
        )
        model = model.to("mps")
        loss = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        return (model, loss, optimizer)
    
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
        
    def get_qs(self, state):
        return self.target_model(np.array(state).reshape(-1, *state.shape)/255)[0]
    
    
    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model(current_states)
        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model(new_current_states)
        
        X = []
        Y = []
        
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
                
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            
            X.append(current_state)
            Y.append(current_qs)
            
        
        fit(self.model, self.optimizer, self.loss, np.array(X)/255, np.array(Y))
        
        # update to determine if we want to update target_model yet
        if terminal_state:
            self.target_update_counter += 1
            
        if self.target_update_counter >= UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0
        
        
                
    
    

def train_one_epoch(model, optimizer, loss_fn, input_data, target_data, epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(zip(input_data, target_data)):
        # Every data instance is an input + label pair
        inputs, labels = data
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(inputs)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(input_data) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    return last_loss



def fit(model, optimizer, loss, input_data, target_data):
    # create the fit method for pytorch
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('dqn_agent_{}'.format(timestamp))
    epoch_number = 0
    EPOCHS = 5
    best_vloss = 1_000_000.
    
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(model, optimizer, loss, input_data, target_data, epoch_number, writer)
        # We don't need gradients on to do reporting
        model.train(False)

        # used for validation data
        running_vloss = 0.0
        for i, vdata in enumerate(zip(input_data, target_data)):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1
        
        
        
agent = DQNAgent()
agent.create_model()

ep_rewards = []
for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episod"):
    episode_reward = 0
    step = 1
    current_state = env.reset()
    
    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)
        new_state, reward, done = env.step(action)
        episode_reward += reward
        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()
            
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)
        
        current_state = new_state
        step += 1
        
    ep_rewards.append(episode_reward)
        
        
            