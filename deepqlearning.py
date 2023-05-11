from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from collections import deque
import numpy as np
import random

OBSERVATION_SPACE_SIZE = 256
ACTION_SPACE_SIZE = 10

REPLAY_MEMORY_SIZE = 50_000
MINIBATCH_SIZE  = 64
MIN_REPLAY_MEMORY_SIZE = 1_000
DISCOUNT = 0.9
UPDATE_TARGET_EVERY = 5


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
        
    def get_qs(self, state, step):
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
        
        
        
# agent = DQNAgent()
# agent.create_model()