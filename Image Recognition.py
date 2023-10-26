# %%
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Define the desired input size for the images
desired_width = 128
desired_height = 128

# Load the CSV file
columns = ["center", "left", "right", "steering", "throttle", "reverse", "speed"]
data = pd.read_csv(r'C:\Users\User\Documents\SELF DRIVING CAR\Driving data.csv', names=columns)

# Extract filenames and labels from the CSV
filenames = data['center'].tolist()
labels = data['steering'].tolist()

# Load and preprocess images (continued)
images = []
for filename in filenames:
    img = cv2.imread(filename)  # No need to add the directory path here
    if img is None:
        print(f"Error loading image: {filename}")
    else:
        img = cv2.resize(img, (desired_width, desired_height))
        img = img / 255.0
        images.append(img)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert the data to NumPy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Check the shapes of the data
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# %%
data.head()

# %%
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the desired input size for the images
desired_width = 128
desired_height = 128

# Load the CSV file
columns = ["center", "left", "right", "steering", "throttle", "reverse", "speed"]
data = pd.read_csv(r'C:\Users\User\Documents\SELF DRIVING CAR\Driving data.csv', names=columns)

# Extract filenames and labels from the CSV
filenames = data['center'].tolist()
labels = data['steering'].tolist()

# Load and preprocess images
images = []
for filename in filenames:
    img = cv2.imread(filename)  # No need to add the directory path here
    if img is None:
        print(f"Error loading image: {filename}")
    else:
        img = cv2.resize(img, (desired_width, desired_height))
        img = img / 255.0
        images.append(img)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,            # Random rotation (degrees)
    width_shift_range=0.1,        # Random horizontal shift
    height_shift_range=0.1,       # Random vertical shift
    brightness_range=[0.8, 1.2],  # Random brightness adjustment
    horizontal_flip=True          # Random horizontal flip
)

# Convert the data to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Apply data augmentation only on the training data
datagen.fit(X_train)


# %%
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the desired input size for the images
desired_width = 128
desired_height = 128

# Load the CSV file
columns = ["center", "left", "right", "steering", "throttle", "reverse", "speed"]
data = pd.read_csv(r'C:\Users\User\Documents\SELF DRIVING CAR\Driving data.csv', names=columns)

# Extract filenames and labels from the CSV
filenames = data['center'].tolist()
labels = data['steering'].tolist()

# Load and preprocess images
images = []
for filename in filenames:
    img = cv2.imread(filename)  # No need to add the directory path here
    if img is None:
        print(f"Error loading image: {filename}")
    else:
        img = cv2.resize(img, (desired_width, desired_height))
        img = img / 255.0
        images.append(img)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,            # Random rotation (degrees)
    width_shift_range=0.1,        # Random horizontal shift
    height_shift_range=0.1,       # Random vertical shift
    brightness_range=[0.8, 1.2],  # Random brightness adjustment
    horizontal_flip=True          # Random horizontal flip
)

# Convert the data to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Apply data augmentation only on the training data
datagen.fit(X_train)

# Check the shapes of the data
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# %%
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Define the desired input size for the images
desired_width = 128
desired_height = 128

# Load the CSV file
columns = ["center", "left", "right", "steering", "throttle", "reverse", "speed"]
data = pd.read_csv(r'C:\Users\User\Documents\SELF DRIVING CAR\Driving data.csv', names=columns)

# Extract filenames and labels from the CSV
filenames = data['center'].tolist()
labels = data['steering'].tolist()

# Load and preprocess images
images = []
for filename in filenames:
    img = cv2.imread(filename)  # No need to add the directory path here
    if img is None:
        print(f"Error loading image: {filename}")
    else:
        img = cv2.resize(img, (desired_width, desired_height))
        img = img / 255.0
        images.append(img)

# Convert the data to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(desired_height, desired_width, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

# Print the model summary
model.summary()


# %%
# Train the model
batch_size = 32
epochs = 10

# Define a ModelCheckpoint callback to save the best model during training
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Train the model
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                    validation_data=(X_test, y_test), callbacks=[checkpoint])

# Save the trained model
model.save('final_model.h5')


# %%
# Load the trained model
from tensorflow.keras.models import load_model

model = load_model('final_model.h5')

# Evaluate the model on the test data
loss = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", loss)


# %%
import matplotlib.pyplot as plt

# Plot the training and validation loss curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# %%
# Extract filenames and labels from the CSV
filenames = data['center'].tolist()
labels = data['steering'].astype(float).tolist()

# %%
# Define the desired input size for the images
desired_width = 128
desired_height = 128

# %%
# Load and preprocess images
images = []
for filename in filenames:
    img = cv2.imread(filename)
    if img is not None:
        img = cv2.resize(img, (desired_width, desired_height))
        img = img / 255.0
        images.append(img)

# %%
# Convert the data to NumPy arrays
X = np.array(images)
y = np.array(labels)

# %%
num_actions = 2

# Define the Deep Q-Network model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(desired_width, desired_height, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_actions, activation='linear'))

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Train the model
model.fit(X, y, batch_size=32, epochs=10, validation_split=0.2)

# Define the Q-table for Q-learning
num_states = 1000  

# %%
import numpy as np

# Hyperparameters for Q-learning
alpha = 0.1
gamma = 0.9
epsilon = 0.1

def select_action(state, q_table):
    if np.random.uniform(0, 1) < epsilon:
        # Select a random action (exploration)
        return np.random.choice(q_table.shape[1])
    else:
        # Select the best action based on Q-values (exploitation)
        return np.argmax(q_table[state, :])

def update_q_table(q_table, state, action, reward, next_state):
    q_predict = q_table[state, action]
    q_target = reward + gamma * np.max(q_table[next_state, :])
    q_table[state, action] += alpha * (q_target - q_predict)

def train_q_learning(data, num_episodes):
    for episode in range(num_episodes):
        # Implementing data loading and preprocessing logic here
        for _, row in data.iterrows():
            state = state_to_int(row['center'], row['left'], row['right'])
            action = select_action(state, q_table)
            reward = row['reward']  

            # Implementing environment step logic here
            # Get the next state and reward based on the action
            next_state = state_to_int(next_center_img, next_left_img, next_right_img)
            next_reward = next_row['reward']

            # Update the Q-table based on the transition
            update_q_table(q_table, state, action, reward, next_state)


def state_to_int(center_image, left_image, right_image):
    # Implementing state representation logic here
    
    return int(hash(center_image + left_image + right_image) % num_states)



# %%
class RLDrivingEnvironment:
    def __init__(self, data):
        self.data = data
        self.current_step = 0
        self.reset()

    def reset(self):
        # Reset the environment to its initial state (starting point of the car)
        self.current_step = 0
        self.current_row = self.data.iloc[self.current_step]

    def step(self, action):
        current_row = self.current_row
        state = state_to_int(current_row['center'], current_row['left'], current_row['right'])

        # Take the action ('action' is an index representing the selected action)
        next_row = self.data.iloc[self.current_step + 1]
        next_center_img, next_left_img, next_right_img = next_row['center'], next_row['left'], next_row['right']
        next_state = state_to_int(next_center_img, next_left_img, next_right_img)

        # Reward is 1 for every step
        reward = 1

        self.current_step += 1
        self.current_row = next_row
        done = (self.current_step == len(self.data) - 1)  # Check if we have reached the end of the data

        return state, reward, next_state, done


# %%
import numpy as np

# Define the number of states and actions based on your data
num_states = 2
num_actions = 2  

# Initialize the Q-table with zeros
q_table = np.zeros((num_states, num_actions))

class RLDrivingEnvironment:
    def __init__(self, data):
        self.data = data
        self.current_step = 0
        self.reset()

    def reset(self):
        # Reset the environment to its initial state (starting point of the car)
        self.current_step = 0
        self.current_row = self.data.iloc[self.current_step]

    def step(self, action):
        current_row = self.current_row
        state = state_to_int(current_row['center'], current_row['left'], current_row['right'])

        # Take the action (assuming 'action' is an index representing the selected action)
        next_row = self.data.iloc[self.current_step + 1]
        next_center_img, next_left_img, next_right_img = next_row['center'], next_row['left'], next_row['right']
        next_state = state_to_int(next_center_img, next_left_img, next_right_img)

        # Reward is 1 for every step
        reward = 1

        self.current_step += 1
        self.current_row = next_row
        done = (self.current_step == len(self.data) - 1)  # Check if we have reached the end of the data

        return state, reward, next_state, done

# Hyperparameters for Q-learning
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Assuming you have a state representation function as follows (concatenate three images into one state)
def state_to_int(center_image, left_image, right_image):
    return int(hash(center_image + left_image + right_image) % num_states)

# Placeholder for the select_action function
def select_action(state, q_table):
    # Implementing  action selection strategy here (e.g., epsilon-greedy)
    pass

# Placeholder for the update_q_table function
def update_q_table(q_table, state, action, reward, next_state):
    # Implementing Q-table update logic here (e.g., Q-learning update)
    pass

# Training
num_episodes = 1000

for episode in range(num_episodes):
    env = RLDrivingEnvironment(data)  # Create a new instance of the environment for each episode
    env.reset()  # Reset the environment to its initial state (starting point of the car)
    state = state_to_int(env.current_row['center'], env.current_row['left'], env.current_row['right'])
    done = False

    while not done:
        action = select_action(state, q_table)
        next_state, reward, next_state, done = env.step(action)
        update_q_table(q_table, state, action, reward, next_state)
        state = next_state


# %%
import numpy as np

# Define the number of states and actions based on your data
num_states = 2  
num_actions = 2  

# Initialize the Q-table with zeros
q_table = np.zeros((num_states, num_actions))

# Hyperparameters for Q-learning
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Training
num_episodes = 1000

for episode in range(num_episodes):
    env = RLDrivingEnvironment(data)  # Create a new instance of the environment for each episode
    env.reset()  # Reset the environment to its initial state (starting point of the car)
    state = state_to_int(env.current_row['center'], env.current_row['left'], env.current_row['right'])
    done = False

    while not done:
        action = select_action(state, q_table)
        next_state, reward, next_state, done = env.step(action)
        update_q_table(q_table, state, action, reward, next_state)
        state = next_state


# %%
import numpy as np

class RLDrivingEnvironment:
    def __init__(self, data):
        self.data = data
        self.current_step = 0
        self.reset()

    def reset(self):
        # Reset the environment to its initial state (starting point of the car)
        self.current_step = 0
        self.current_row = self.data.iloc[self.current_step]

    def step(self, action):
        current_row = self.current_row
        state = state_to_int(current_row['center'], current_row['left'], current_row['right'])

        # Take the action ('action' is an index representing the selected action)
        next_row = self.data.iloc[self.current_step + 1]
        next_center_img, next_left_img, next_right_img = next_row['center'], next_row['left'], next_row['right']
        next_state = state_to_int(next_center_img, next_left_img, next_right_img)

        # Reward is 1 for every step
        reward = 1

        self.current_step += 1
        self.current_row = next_row
        done = (self.current_step == len(self.data) - 1)  # Check if we have reached the end of the data

        return state, reward, next_state, done

# Hyperparameters for Q-learning
alpha = 0.1
gamma = 0.9
epsilon = 0.1


def state_to_int(center_image, left_image, right_image):
    return int(hash(center_image + left_image + right_image) % num_states)

# Training
num_episodes = 1000

for episode in range(num_episodes):
    env = RLDrivingEnvironment(data)  # Create a new instance of the environment for each episode
    env.reset()  # Reset the environment to its initial state (starting point of the car)
    state = state_to_int(env.current_row['center'], env.current_row['left'], env.current_row['right'])
    done = False

    while not done:
        action = select_action(state, q_table)
        next_state, reward, next_state, done = env.step(action)
        update_q_table(q_table, state, action, reward, next_state)
        state = next_state


# %%
import numpy as np


def state_to_int(center_image, left_image, right_image):
    return int(hash(center_image + left_image + right_image) % num_states)

# Define the action space
actions = ['center', 'left', 'right']
num_actions = len(actions)

# Define the Q-table for Q-learning (you can adjust the number of states)
num_states = 1000  
q_table = np.zeros((num_states, num_actions))

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Define the select_action function
def select_action(state, q_table):
    if np.random.uniform(0, 1) < epsilon:
        # Select a random action (exploration)
        return np.random.choice(num_actions)
    else:
        # Select the best action based on Q-values (exploitation)
        return np.argmax(q_table[state, :])

# Define the update_q_table function
def update_q_table(q_table, state, action, reward, next_state):
    q_predict = q_table[state, action]
    q_target = reward + gamma * np.max(q_table[next_state, :])
    q_table[state, action] += alpha * (q_target - q_predict)

# Training
num_episodes = 1000

for episode in range(num_episodes):
    env = RLDrivingEnvironment(data)  # Create a new instance of the environment for each episode
    env.reset()  # Reset the environment to its initial state (starting point of the car)
    state = state_to_int(env.current_row['center'], env.current_row['left'], env.current_row['right'])
    done = False

    while not done:
        action = select_action(state, q_table)
        next_state, reward, next_state, done = env.step(action)
        update_q_table(q_table, state, action, reward, next_state)
        state = next_state

# Q-learning is now complete, and the Q-table is updated.



# %%
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Define the desired input size for the images
desired_width = 128
desired_height = 128


def state_to_int(center_image, left_image, right_image):
    return int(hash(center_image + left_image + right_image) % num_states)

# Load the CSV file
columns = ["center", "left", "right", "steering", "throttle", "reverse", "speed"]
data = pd.read_csv(r'C:\Users\User\Documents\SELF DRIVING CAR\Driving data.csv', names=columns)

# Preprocess the image data
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (desired_width, desired_height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return image

# Split your data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2)  

# Define the action space
actions = ['center', 'left', 'right']
num_actions = len(actions)

# Define the Q-table for Q-learning (you can adjust the number of states)
num_states = 1000  
q_table = np.zeros((num_states, num_actions))

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Define the select_action function
def select_action(state, q_table):
    if np.random.uniform(0, 1) < epsilon:
        # Select a random action (exploration)
        return np.random.choice(num_actions)
    else:
        # Select the best action based on Q-values (exploitation)
        return np.argmax(q_table[state, :])

# Define the update_q_table function
def update_q_table(q_table, state, action, reward, next_state):
    q_predict = q_table[state, action]
    q_target = reward + gamma * np.max(q_table[next_state, :])
    q_table[state, action] += alpha * (q_target - q_predict)

# Training
num_episodes = 1000

for episode in range(num_episodes):
    env = RLDrivingEnvironment(train_data)  # Use the training data for training the Q-learning agent
    env.reset()
    state = state_to_int(env.current_row['center'], env.current_row['left'], env.current_row['right'])
    done = False

    while not done:
        action = select_action(state, q_table)
        next_state, reward, next_state, done = env.step(action)
        update_q_table(q_table, state, action, reward, next_state)
        state = next_state

# Q-learning is now complete, and the Q-table is updated.

# Now, we have to evaluate the performance on the test dataset
test_env = RLDrivingEnvironment(test_data)
test_env.reset()
test_state = state_to_int(test_env.current_row['center'], test_env.current_row['left'], test_env.current_row['right'])
test_done = False

test_actions = []  # To store the actions taken during the test episode
test_rewards = []  # To store the rewards received during the test episode

while not test_done:
    test_action = select_action(test_state, q_table)  # Corrected function call
    test_actions.append(test_action)  # Store the action taken
    test_state, test_reward, next_state, test_done = test_env.step(test_action)  # Unpack the correct values here
    test_rewards.append(test_reward)  # Store the reward received

# Print the output of the test episode
print("Test Actions:", test_actions)
print("Test Rewards:", test_rewards)



# %%
# Calculate metrics for the test episode
total_reward = sum(test_rewards)
average_reward_per_step = total_reward / len(test_rewards)
success_rate = 1 if test_done else 0  # If test_done is True, it means the episode was successful
action_distribution = {action: test_actions.count(action) for action in actions}

# Print the results
print("Test Episode Results:")
print("Total Reward:", total_reward)
print("Average Reward per Step:", average_reward_per_step)
print("Success Rate:", success_rate)
print("Action Distribution:", action_distribution)


# %%
import matplotlib.pyplot as plt
import numpy as np

# Visualize the rewards obtained during the test episode using a line plot
plt.figure()
plt.plot(test_rewards)
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Test Episode Reward Plot')
plt.show()

# Define the action space manually (replace the action labels with your actual action names)
action_space = ['left', 'center', 'right']

# Plot the histogram of action distribution
plt.figure()
plt.hist(test_actions, bins=np.arange(len(action_space) + 1) - 0.5, rwidth=0.8, align='mid')
plt.xticks(range(len(action_space)), action_space)
plt.xlabel('Action')
plt.ylabel('Frequency')
plt.title('Test Episode Action Distribution')
plt.show()


# %%
import gym
from gym import spaces
import numpy as np

class SelfDrivingCarEnv(gym.Env):
    def __init__(self, data):
        super(SelfDrivingCarEnv, self).__init__()

        # Define the action and observation spaces here
        self.action_space = spaces.Discrete(len(action_space))
        self.observation_space = spaces.Box(low=0, high=255, shape=(desired_height, desired_width, 3), dtype=np.uint8)  

        # Initialize other variables and environment-specific parameters here
        self.data = data
        self.current_step = 0
        self.current_row = self.data.iloc[self.current_step]

    def reset(self):
        # Reset the environment to its initial state (starting point of the car)
        self.current_step = 0
        self.current_row = self.data.iloc[self.current_step]

    def step(self, action):
        current_row = self.current_row
        state = preprocess_image(current_row['center'])  
        reward = 1  
        # Take the action ('action' is an index representing the selected action)
        next_row = self.data.iloc[self.current_step + 1]
        next_center_img = preprocess_image(next_row['center'])  

        self.current_step += 1
        self.current_row = next_row
        done = (self.current_step == len(self.data) - 1)  

        return state, reward, done, {}  # Return state, reward, done, and additional info

    def render(self, mode='human')
        pass


# %%
import gym
from gym import spaces
import numpy as np

class SimpleDrivingEnv(gym.Env):
    def __init__(self):
        super(SimpleDrivingEnv, self).__init__()

        # Define action space (3 actions: left, straight, right)
        self.action_space = spaces.Discrete(3)

        # Define observation space (2D position of the car)
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        # Initialize the car's position
        self.car_position = np.array([0.5, 0.1])

        # Define road parameters (in this example, a simple straight road)
        self.road_width = 0.2
        self.road_center = 0.5

    def reset(self):
        # Reset the car's position to the starting point
        self.car_position = np.array([0.5, 0.1])
        return self.car_position

    def step(self, action):
        # Define how the car moves based on the selected action
        if action == 0:  # Left
            self.car_position[0] -= 0.05
        elif action == 1:  # Straight (center)
            pass  # Car stays in the current lane
        elif action == 2:  # Right
            self.car_position[0] += 0.05

        # Simulate the car's movement
        self.car_position[1] += 0.05  # Car moves forward

        # Calculate reward based on the car's position
        reward = self.calculate_reward()

        # Check if the car has reached the end of the road
        done = self.car_position[1] >= 1.0

        # Return the current observation, reward, whether it's done, and additional info
        return self.car_position, reward, done, {}

    def calculate_reward(self):
        # Calculate reward based on the car's position relative to the road center
        distance_to_center = abs(self.car_position[0] - self.road_center)
        if distance_to_center < self.road_width / 2:
            return 1.0  # Maximum reward when the car is in the center of the road
        else:
            return 0.0  # No reward when the car is outside the road

    def render(self, mode='human'):
        # Visualize the environment 
        pass  

    def close(self):
        pass

# Create the environment
env = SimpleDrivingEnv()

# Main training loop
for episode in range(1000):
    observation = env.reset()
    total_reward = 0

    while True:
        action = env.action_space.sample()

        observation, reward, done, _ = env.step(action)
        total_reward += reward

        env.render()

        if done:
            break

env.close()


# %%
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define a custom environment (you can replace this with your RL environment)
class CustomEnv:
    def __init__(self):
        self.state = 0

    def step(self, action):
        self.state += action
        reward = 1 if self.state >= 5 else 0
        done = self.state >= 10
        return self.state, reward, done, {}

    def reset(self):
        self.state = 0
        return self.state

# Create the environment
env = CustomEnv()

# Set up the visualization
fig, ax = plt.subplots()
states = []
rewards = []

# Function to update the visualization
def update(frame):
    states.append(env.state)
    rewards.append(env.step(np.random.choice([1, 2]))[1])
    ax.clear()
    ax.plot(states, label='State')
    ax.plot(rewards, label='Reward')
    ax.legend()

# Create the animation
ani = FuncAnimation(fig, update, frames=range(100), repeat=False)

# Save the animation as a GIF
ani.save('learning_progress.gif', writer='pillow', fps=10)

plt.show()


# %%
import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt


class SimpleDrivingEnv(gym.Env):
    def __init__(self):
        super(SimpleDrivingEnv, self).__init__()

        # Define action space (3 actions: left, straight, right)
        self.action_space = spaces.Discrete(3)

        # Define observation space (2D position of the car)
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        # Initialize the car's position
        self.car_position = np.array([0.5, 0.1])

        # Define road parameters (in this example, a simple straight road)
        self.road_width = 0.2
        self.road_center = 0.5

    def reset(self):
        # Reset the car's position to the starting point
        self.car_position = np.array([0.5, 0.1])
        return self.car_position

    def step(self, action):
        # Define how the car moves based on the selected action
        if action == 0:  # Left
            self.car_position[0] -= 0.05
        elif action == 1:  # Straight (center)
            pass  # Car stays in the current lane
        elif action == 2:  # Right
            self.car_position[0] += 0.05

        # Simulate the car's movement
        self.car_position[1] += 0.05  # Car moves forward

        # Calculate reward based on the car's position
        reward = self.calculate_reward()

        # Check if the car has reached the end of the road
        done = self.car_position[1] >= 1.0

        # Return the current observation, reward, whether it's done, and additional info
        return self.car_position, reward, done, {}

    def calculate_reward(self):
        # Calculate reward based on the car's position relative to the road center
        distance_to_center = abs(self.car_position[0] - self.road_center)
        if distance_to_center < self.road_width / 2:
            return 1.0  # Maximum reward when the car is in the center of the road
        else:
            return 0.0  # No reward when the car is outside the road

    def render(self, mode='human'):
        if mode == 'human':
            plt.clf()
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.gca().add_patch(plt.Rectangle((self.road_center - self.road_width / 2, 0), self.road_width, 1, color='gray'))
            plt.gca().add_patch(plt.Rectangle((self.car_position[0] - 0.025, self.car_position[1] - 0.025), 0.05, 0.05, color='blue'))
            plt.pause(0.01)
        elif mode == 'rgb_array':
            raise NotImplementedError("Rendering as RGB array is not supported in this example.")
        

# Create the environment
env = SimpleDrivingEnv()

# Create a video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
video_filename = 'learning_animation.mp4'
video_writer = cv2.VideoWriter(video_filename, fourcc, 25, (640, 480))  # 25 FPS, frame size (640x480)

# Main training loop
for episode in range(1000):
    observation = env.reset()
    total_reward = 0

    while True:
        action = env.action_space.sample()

        observation, reward, done, _ = env.step(action)
        total_reward += reward

        # Visualization
        plt.clf()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().add_patch(plt.Rectangle((env.road_center - env.road_width / 2, 0), env.road_width, 1, color='gray'))
        plt.gca().add_patch(plt.Rectangle((observation[0] - 0.025, observation[1] - 0.025), 0.05, 0.05, color='blue'))
        plt.pause(0.01)

        # Capture the current figure as an image and convert it to a NumPy array
        fig = plt.gcf()
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())

        # Resize the frame to the desired size (640x480 in this example)
        frame = cv2.resize(frame, (640, 480))

        # Write the frame to the video file
        video_writer.write(frame)

        if done:
            break

# Release the video writer
video_writer.release()

print(f"Video saved as {video_filename}")     

# %%
import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import subprocess

class SimpleDrivingEnv(gym.Env):
    def __init__(self):
        super(SimpleDrivingEnv, self).__init__()

        # Define action space (3 actions: left, straight, right)
        self.action_space = gym.spaces.Discrete(3)

        # Define observation space (2D position of the car)
        self.observation_space = gym.spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        # Initialize the car's position
        self.car_position = np.array([0.5, 0.1])

        # Define road parameters (in this example, a simple straight road)
        self.road_width = 0.2
        self.road_center = 0.5

    def reset(self):
        # Reset the car's position to the starting point
        self.car_position = np.array([0.5, 0.1])
        return self.car_position

    def step(self, action):
        # Define how the car moves based on the selected action
        if action == 0:  # Left
            self.car_position[0] -= 0.05
        elif action == 1:  # Straight (center)
            pass  # Car stays in the current lane
        elif action == 2:  # Right
            self.car_position[0] += 0.05

        # Simulate the car's movement
        self.car_position[1] += 0.05  # Car moves forward

        # Calculate reward based on the car's position
        reward = self.calculate_reward()

        # Check if the car has reached the end of the road
        done = self.car_position[1] >= 1.0

        # Return the current observation, reward, whether it's done, and additional info
        return self.car_position, reward, done, {}

    def calculate_reward(self):
        # Calculate reward based on the car's position relative to the road center
        distance_to_center = abs(self.car_position[0] - self.road_center)
        if distance_to_center < self.road_width / 2:
            return 1.0  # Maximum reward when the car is in the center of the road
        else:
            return 0.0  # No reward when the car is outside the road

    def render(self, mode='human'):
        if mode == 'human':
            plt.clf()
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.gca().add_patch(plt.Rectangle((self.road_center - self.road_width / 2, 0), self.road_width, 1, color='gray'))
            plt.gca().add_patch(plt.Rectangle((self.car_position[0] - 0.025, self.car_position[1] - 0.025), 0.05, 0.05, color='blue'))
            plt.pause(0.01)
        elif mode == 'rgb_array':
            raise NotImplementedError("Rendering as RGB array is not supported in this example.")

# Create the environment
env = SimpleDrivingEnv()

# Create a directory to save captured frames
output_frames_dir = r'C:\Users\User\Documents\SELF DRIVING CAR\output_frames'

os.makedirs(output_frames_dir, exist_ok=True)

# Create a list to store captured frame filenames
captured_frames = []

for episode in range(1000):
    observation = env.reset()
    total_reward = 0

    while True:
        action = env.action_space.sample()

        observation, reward, done, _ = env.step(action)
        total_reward += reward

        # Visualization
        plt.clf()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().add_patch(plt.Rectangle((env.road_center - env.road_width / 2, 0), env.road_width, 1, color='gray'))
        plt.gca().add_patch(plt.Rectangle((observation[0] - 0.025, observation[1] - 0.025), 0.05, 0.05, color='blue'))

        # Capture the current figure as an image and convert it to a NumPy array
        fig = plt.gcf()
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())

        # Resize the frame to the desired size (640x480 in this example)
        frame = cv2.resize(frame, (640, 480))

        # Write the frame to the video file
        video_writer.write(frame)

        if done:
            break


# After the loop, captured_frames will contain the filenames of the captured frames.

# Create a video from the captured frames using FFmpeg
output_video_filename = 'learning_animation.mp4'
frame_rate = 50  # Adjust the frame rate as needed

# Run FFmpeg to create the video
ffmpeg_cmd = f"ffmpeg -r {frame_rate} -i {output_frames_dir}/frame_%04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p {output_video_filename}"
subprocess.run(ffmpeg_cmd, shell=True)

# Optionally, you can delete the individual frame image files to save space
for frame_filename in captured_frames:
    os.remove(frame_filename)

print(f"Video saved as {output_video_filename}")


# %%
import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import subprocess

class SimpleDrivingEnv(gym.Env):
    def __init__(self):
        super(SimpleDrivingEnv, self).__init__()

        # Define action space (3 actions: left, straight, right)
        self.action_space = gym.spaces.Discrete(3)

        # Define observation space (2D position of the car)
        self.observation_space = gym.spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        # Initialize the car's position
        self.car_position = np.array([0.5, 0.1])

        # Define road parameters (in this example, a simple straight road)
        self.road_width = 0.2
        self.road_center = 0.5

    def reset(self):
        # Reset the car's position to the starting point
        self.car_position = np.array([0.5, 0.1])
        return self.car_position

    def step(self, action):
        # Define how the car moves based on the selected action
        if action == 0:  # Left
            self.car_position[0] -= 0.05
        elif action == 1:  # Straight (center)
            pass  # Car stays in the current lane
        elif action == 2:  # Right
            self.car_position[0] += 0.05

        # Simulate the car's movement
        self.car_position[1] += 0.05  # Car moves forward

        # Calculate reward based on the car's position
        reward = self.calculate_reward()

        # Check if the car has reached the end of the road
        done = self.car_position[1] >= 1.0

        # Return the current observation, reward, whether it's done, and additional info
        return self.car_position, reward, done, {}

    def calculate_reward(self):
        # Calculate reward based on the car's position relative to the road center
        distance_to_center = abs(self.car_position[0] - self.road_center)
        if distance_to_center < self.road_width / 2:
            return 1.0  # Maximum reward when the car is in the center of the road
        else:
            return 0.0  # No reward when the car is outside the road

# Create the environment
env = SimpleDrivingEnv()

# Create a directory to save captured frames
output_frames_dir = r'C:\Users\User\Documents\SELF DRIVING CAR\output_frames'
os.makedirs(output_frames_dir, exist_ok=True)

# Create a list to store captured frame filenames
captured_frames = []

# Create a Matplotlib figure and axis outside the loop
fig, ax = plt.subplots(figsize=(6.4, 4.8))

# Create a video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
output_video_filename = r'C:\Users\User\Documents\SELF DRIVING CAR\learning_animation.mp4'
frame_rate = 50  # Adjust the frame rate as needed
video_writer = cv2.VideoWriter(output_video_filename, fourcc, frame_rate, (640, 480))

# Main training loop
for episode in range(1000):
    observation = env.reset()
    total_reward = 0

    while True:
        action = env.action_space.sample()

        observation, reward, done, _ = env.step(action)
        total_reward += reward

        # Clear the axis for the new frame
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.add_patch(plt.Rectangle((env.road_center - env.road_width / 2, 0), env.road_width, 1, color='gray'))
        ax.add_patch(plt.Rectangle((observation[0] - 0.025, observation[1] - 0.025), 0.05, 0.05, color='blue'))

        # Capture the current figure as an image and convert it to a NumPy array
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())

        # Resize the frame to the desired size (640x480 in this example)
        frame = cv2.resize(frame, (640, 480))

        # Write the frame to the video file
        video_writer.write(frame)

        if done:
            break

# Release the video writer
video_writer.release()

print(f"Video saved as {output_video_filename}")

# %%
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2

class SimpleDrivingEnv(gym.Env):
    def __init__(self):
        super(SimpleDrivingEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.car_position = np.array([0.5, 0.1])
        self.road_width = 0.2
        self.road_center = 0.5

    def reset(self):
        self.car_position = np.array([0.5, 0.1])
        return self.car_position

    def step(self, action):
        if action == 0:
            self.car_position[0] -= 0.05
        elif action == 1:
            pass
        elif action == 2:
            self.car_position[0] += 0.05

        self.car_position[1] += 0.05
        reward = self.calculate_reward()
        done = self.car_position[1] >= 1.0
        return self.car_position, reward, done, {}

    def calculate_reward(self):
        distance_to_center = abs(self.car_position[0] - self.road_center)
        if distance_to_center < self.road_width / 2:
            return 1.0
        else:
            return 0.0

# Create the environment
env = SimpleDrivingEnv()

# Create a directory to save captured frames
output_frames_dir = 'output_frames'
os.makedirs(output_frames_dir, exist_ok=True)

# Create a Matplotlib figure and axis
fig, ax = plt.subplots(figsize=(6.4, 4.8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
road = ax.add_patch(plt.Rectangle((env.road_center - env.road_width / 2, 0), env.road_width, 1, color='gray'))
car = ax.add_patch(plt.Rectangle((0.5 - 0.025, 0.1 - 0.025), 0.05, 0.05, color='blue'))

# Frame rate control (increase or decrease as needed)
frame_rate = 5

# Main loop for visualization
for episode in range(1000):
    observation = env.reset()
    total_reward = 0

    while True:
        action = env.action_space.sample()

        observation, reward, done, _ = env.step(action)
        total_reward += reward

        car.set_xy((observation[0] - 0.025, observation[1] - 0.025))

        # Capture the current figure as an image and convert it to a NumPy array
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())

        # Resize the frame to the desired size (640x480 in this example)
        frame = cv2.resize(frame, (640, 480))

        # Save the frame as an image
        frame_filename = os.path.join(output_frames_dir, f'frame_{episode:04d}.png')
        cv2.imwrite(frame_filename, frame)

        if done:
            break

print(f"Frames saved in the directory: {output_frames_dir}")

# %%
print("Total Data:", len(data))

# %%
def load_img_steering(datadir, df):
    image_path = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
        image_path.append(os.path.join(datadir, center.strip()))
        steering.append(float(indexed_data[3]))
        image_path.append(os.path.join(datadir, left.strip()))
        steering.append(float(indexed_data[3]) + 0.15)
        image_path.append(os.path.join(datadir, right.strip()))
        steering.append(float(indexed_data[3]) - 0.15)
    image_paths = np.asarray(image_path)
    steerings = np.asarray(steering)
    return image_paths, steerings

# %%
import os
image_paths, steerings = load_img_steering(r'C:\Users\User\Documents\SELF DRIVING CAR\Driving data.csv', data)
X_train, X_valid, y_train, y_valid = train_test_split(
    image_paths, steerings, test_size=0.2, random_state=6
)
print("Training Samples: {}\nValid Samples: {}".format(len(X_train), len(X_valid)))


# %%

# Define num_bins
num_bins = 25
# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot histograms
axes[0].hist(y_train, bins=num_bins, width=0.05, color="blue")
axes[0].set_title("Training set")

axes[1].hist(y_valid, bins=num_bins, width=0.05, color="red")
axes[1].set_title("Validation set")

plt.show()

# %% [markdown]
# #COMPARING MY DATASET WITH THE OTHER DATASET

# %%
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

# Define the input size for the images
desired_width = 128
desired_height = 128

# Define a function to load, preprocess, and split the data
def preprocess_and_split_data(dataset_path):
    # Preprocessing steps may include loading images, resizing, normalizing, etc.
    data = pd.read_csv(dataset_path, names=["center", "left", "right", "steering", "throttle", "reverse", "speed"])

    images = []  # To store preprocessed images
    labels = []  # To store corresponding labels

    for index, row in data.iterrows():
        # Load and preprocess image
        img = cv2.imread(row['center'])  
        img = cv2.resize(img, (desired_width, desired_height))
        img = img / 255.0  # Normalize the pixel values
        images.append(img)

        
        label = float(row['steering'])  # Convert the steering angle to float
        labels.append(label)

    # Convert lists to NumPy arrays
    X = np.array(images)
    y = np.array(labels)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# my dataset paths
my_dataset_path = r'C:\Users\User\Documents\SELF DRIVING CAR\Driving data.csv'
another_dataset_path = r'C:\Users\User\Documents\Comparing\driving_log.csv'

# Load and preprocess "my Dataset"
X_train_my, X_test_my, y_train_my, y_test_my = preprocess_and_split_data(my_dataset_path)

# Load and preprocess "Another Dataset"
X_train_another, X_test_another, y_train_another, y_test_another = preprocess_and_split_data(another_dataset_path)


# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Train a regression model on "My Dataset"
model_my = LinearRegression()
model_my.fit(X_train_my.reshape(-1, desired_width * desired_height * 3), y_train_my)

# Predict on the test set from "My Dataset"
y_pred_my = model_my.predict(X_test_my.reshape(-1, desired_width * desired_height * 3))

# Calculate MSE for "My Dataset"
mse_my = mean_squared_error(y_test_my, y_pred_my)
print(f"Mean Squared Error for My Dataset: {mse_my}")

# Train a regression model on "Another Dataset"
model_another = LinearRegression()
model_another.fit(X_train_another.reshape(-1, desired_width * desired_height * 3), y_train_another)

# Predict on the test set from "Another Dataset"
y_pred_another = model_another.predict(X_test_another.reshape(-1, desired_width * desired_height * 3))

# Calculate MSE for "Another Dataset"
mse_another = mean_squared_error(y_test_another, y_pred_another)
print(f"Mean Squared Error for Another Dataset: {mse_another}")


# %%
import matplotlib.pyplot as plt

# Visualize predictions on "My Dataset" with colorful lines
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(y_test_my, label='Actual Steering Angle', color='blue')
plt.plot(y_pred_my, label='Predicted Steering Angle', color='red', linestyle='--')
plt.fill_between(range(len(y_test_my)), y_test_my, y_pred_my, where=(y_test_my >= y_pred_my), interpolate=True, color='green', alpha=0.5)
plt.fill_between(range(len(y_test_my)), y_test_my, y_pred_my, where=(y_test_my < y_pred_my), interpolate=True, color='orange', alpha=0.5)
plt.title('Predictions on My Dataset')
plt.xlabel('Sample')
plt.ylabel('Steering Angle')
plt.legend()

# Visualize predictions on "Another Dataset" with colorful lines
plt.subplot(2, 1, 2)
plt.plot(y_test_another, label='Actual Steering Angle', color='blue')
plt.plot(y_pred_another, label='Predicted Steering Angle', color='red', linestyle='--')
plt.fill_between(range(len(y_test_another)), y_test_another, y_pred_another, where=(y_test_another >= y_pred_another), interpolate=True, color='green', alpha=0.5)
plt.fill_between(range(len(y_test_another)), y_test_another, y_pred_another, where=(y_test_another < y_pred_another), interpolate=True, color='orange', alpha=0.5)
plt.title('Predictions on Another Dataset')
plt.xlabel('Sample')
plt.ylabel('Steering Angle')
plt.legend()

plt.tight_layout()
plt.show()


# %%
from sklearn.metrics import mean_squared_error
from math import sqrt

# Calculate Mean Squared Error (MSE) for "my Dataset"
mse_my = mean_squared_error(y_test_my, y_pred_my)

# Calculate Root Mean Squared Error (RMSE) for "Your Dataset"
rmse_my = sqrt(mse_my)

# Calculate Mean Squared Error (MSE) for "Another Dataset"
mse_another = mean_squared_error(y_test_another, y_pred_another)

# Calculate Root Mean Squared Error (RMSE) for "Another Dataset"
rmse_another = sqrt(mse_another)

# Print the MSE and RMSE for both datasets
print("MSE for My Dataset:", mse_my)
print("RMSE for My Dataset:", rmse_my)
print("MSE for Another Dataset:", mse_another)
print("RMSE for Another Dataset:", rmse_another)


# %%
import matplotlib.pyplot as plt

# Calculate prediction errors for both datasets
error_my = y_test_my - y_pred_my
error_another = y_test_another - y_pred_another

# Plot error distribution histogram for "My Dataset"
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.hist(error_my, bins=50, color='blue', alpha=0.7)
plt.title('Error Distribution for My Dataset')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')

# Plot error distribution histogram for "Another Dataset"
plt.subplot(2, 1, 2)
plt.hist(error_another, bins=50, color='red', alpha=0.7)
plt.title('Error Distribution for Another Dataset')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# %%
from scipy import stats

# Perform a t-test to compare prediction errors between the two datasets
t_statistic, p_value = stats.ttest_ind(error_my, error_another)

# Check if the p-value is significant (e.g., less than 0.05)
if p_value < 0.05:
    print("The difference in prediction errors between the two datasets is statistically significant.")
else:
    print("There is no statistically significant difference in prediction errors between the two datasets.")


# %%
import matplotlib.pyplot as plt

# Create box plots for prediction errors
plt.figure(figsize=(10, 6))

# Box plot for "My Dataset"
plt.subplot(2, 1, 1)
plt.boxplot(error_my, vert=False, patch_artist=True, boxprops=dict(facecolor='blue', alpha=0.7))
plt.title('Box Plot of Prediction Errors for My Dataset')
plt.xlabel('Prediction Error')

# Box plot for "Another Dataset"
plt.subplot(2, 1, 2)
plt.boxplot(error_another, vert=False, patch_artist=True, boxprops=dict(facecolor='red', alpha=0.7))
plt.title('Box Plot of Prediction Errors for Another Dataset')
plt.xlabel('Prediction Error')

plt.tight_layout()
plt.show()


# %%
from sklearn.metrics import mean_squared_error

# Calculate RMSE for "My Dataset"
rmse_my = np.sqrt(mean_squared_error(y_test_my, y_pred_my))

# Calculate RMSE for "Another Dataset"
rmse_another = np.sqrt(mean_squared_error(y_test_another, y_pred_another))

# Print the RMSE values
print(f"RMSE for My Dataset: {rmse_my:.4f}")
print(f"RMSE for Another Dataset: {rmse_another:.4f}")


# %%
from sklearn.metrics import mean_absolute_error

# Calculate Mean Absolute Error (MAE) for "My Dataset"
mae_my = mean_absolute_error(y_test_my, y_pred_my)

# Calculate Mean Absolute Error (MAE) for "Another Dataset"
mae_another = mean_absolute_error(y_test_another, y_pred_another)

# Print the MAE values
print(f"Mean Absolute Error (MAE) for My Dataset: {mae_my:.4f}")
print(f"Mean Absolute Error (MAE) for Another Dataset: {mae_another:.4f}")


# %%
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define the input size for the images
desired_width = 128
desired_height = 128

# Define the steering angle categories and corresponding labels
categories = {
    "left_turn": (-1.0, -0.3),
    "straight": (-0.3, 0.3),
    "right_turn": (0.3, 1.0)
}

# Define a function to convert regression labels to classification labels
def convert_to_classification_labels(steering_angle, categories):
    for label, (min_val, max_val) in categories.items():
        if min_val <= steering_angle <= max_val:
            return label
    return None  # Handle cases outside defined categories

# Define a function to load, preprocess, and split the data
def preprocess_and_split_data(dataset_path, categories):
    data = pd.read_csv(dataset_path, names=["center", "left", "right", "steering", "throttle", "reverse", "speed"])

    images = []  # To store preprocessed images
    labels = []  # To store corresponding classification labels

    for index, row in data.iterrows():
        # Load and preprocess image
        img = cv2.imread(row['center'])  # Assuming the 'center' column contains image paths
        img = cv2.resize(img, (desired_width, desired_height))
        img = img / 255.0  # Normalize the pixel values
        images.append(img)

        # Convert and preprocess label (steering angle to classification label)
        label = float(row['steering'])  # Convert the steering angle to float
        classification_label = convert_to_classification_labels(label, categories)
        labels.append(classification_label)

    # Convert lists to NumPy arrays
    X = np.array(images)
    y = np.array(labels)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Define your dataset paths
my_dataset_path = r'C:\Users\User\Documents\SELF DRIVING CAR\Driving data.csv'
another_dataset_path = r'C:\Users\User\Documents\Comparing\driving_log.csv'

# Load and preprocess the first dataset
X_train_my, X_test_my, y_train_my, y_test_my = preprocess_and_split_data(my_dataset_path, categories)

# Load and preprocess the second dataset
X_train_another, X_test_another, y_train_another, y_test_another = preprocess_and_split_data(another_dataset_path, categories)

# Create a Logistic Regression classifier (you can use a different classifier as needed)
classifier = LogisticRegression(max_iter=1000)

# Flatten the image data for the classifier
X_train_flat_my = X_train_my.reshape(X_train_my.shape[0], -1)
X_test_flat_my = X_test_my.reshape(X_test_my.shape[0], -1)

X_train_flat_another = X_train_another.reshape(X_train_another.shape[0], -1)
X_test_flat_another = X_test_another.reshape(X_test_another.shape[0], -1)

# Train the classifier on the first dataset
classifier.fit(X_train_flat_my, y_train_my)

# Predict on the test set from the first dataset
y_pred_my = classifier.predict(X_test_flat_my)

# Calculate accuracy and classification report for the first dataset
accuracy_my = accuracy_score(y_test_my, y_pred_my)
class_report_my = classification_report(y_test_my, y_pred_my)

# Train the classifier on the second dataset
classifier.fit(X_train_flat_another, y_train_another)

# Predict on the test set from the second dataset
y_pred_another = classifier.predict(X_test_flat_another)

# Calculate accuracy and classification report for the second dataset
accuracy_another = accuracy_score(y_test_another, y_pred_another)
class_report_another = classification_report(y_test_another, y_pred_another)

# Print the results for both datasets
print("Results for My Dataset:")
print(f"Accuracy: {accuracy_my:.4f}")
print("Classification Report:\n", class_report_my)

print("\nResults for Another Dataset:")
print(f"Accuracy: {accuracy_another:.4f}")
print("Classification Report:\n", class_report_another)

# Generate confusion matrices
cm_my = confusion_matrix(y_test_my, y_pred_my)
cm_another = confusion_matrix(y_test_another, y_pred_another)

# Define class labels
class_labels = ['left_turn', 'straight', 'right_turn']

# Plot confusion matrix for My Dataset
plt.figure(figsize=(8, 6))
sns.heatmap(cm_my, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for My Dataset')
plt.show()

# Plot confusion matrix for Another Dataset
plt.figure(figsize=(8, 6))
sns.heatmap(cm_another, annot=True, fmt="d", cmap="Reds", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Another Dataset')
plt.show()


# %% [markdown]
# COMPARING THE ANOTHER DATASET ON THE TRACK

# %%
import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt


class SimpleDrivingEnv(gym.Env):
    def __init__(self):
        super(SimpleDrivingEnv, self).__init__()

        # Define action space (3 actions: left, straight, right)
        self.action_space = spaces.Discrete(3)

        # Define observation space (2D position of the car)
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        # Initialize the car's position
        self.car_position = np.array([0.5, 0.1])

        # Define road parameters ,a simple straight road
        self.road_width = 0.2
        self.road_center = 0.5

    def reset(self):
        # Reset the car's position to the starting point
        self.car_position = np.array([0.5, 0.1])
        return self.car_position

    def step(self, action):
        # Define how the car moves based on the selected action
        if action == 0:  # Left
            self.car_position[0] -= 0.05
        elif action == 1:  # Straight (center)
            pass  # Car stays in the current lane
        elif action == 2:  # Right
            self.car_position[0] += 0.05

        # Simulate the car's movement
        self.car_position[1] += 0.05  # Car moves forward

        # Calculate reward based on the car's position
        reward = self.calculate_reward()

        # Check if the car has reached the end of the road
        done = self.car_position[1] >= 1.0

        # Return the current observation, reward, whether it's done, and additional info
        return self.car_position, reward, done, {}

    def calculate_reward(self):
        # Calculate reward based on the car's position relative to the road center
        distance_to_center = abs(self.car_position[0] - self.road_center)
        if distance_to_center < self.road_width / 2:
            return 1.0  # Maximum reward when the car is in the center of the road
        else:
            return 0.0  # No reward when the car is outside the road

    def render(self, mode='human'):
        if mode == 'human':
            plt.clf()
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.gca().add_patch(plt.Rectangle((self.road_center - self.road_width / 2, 0), self.road_width, 1, color='gray'))
            plt.gca().add_patch(plt.Rectangle((self.car_position[0] - 0.025, self.car_position[1] - 0.025), 0.05, 0.05, color='blue'))
            plt.pause(0.01)
        elif mode == 'rgb_array':
            raise NotImplementedError("Rendering as RGB array is not supported in this example.")
        
# Directory to save visualized images
image_directory = 'C:/Users/User/Documents/SELF DRIVING CAR/Car learning image'

# Create the environment
env = SimpleDrivingEnv()

# Main training loop
for episode in range(1000):
    observation = env.reset()
    total_reward = 0

    episode_directory = os.path.join(image_directory, f'episode_{episode}')
    os.makedirs(episode_directory, exist_ok=True)

    frame_count = 0  # To keep track of frames within the episode

    while True:
        action = env.action_space.sample()

        observation, reward, done, _ = env.step(action)
        total_reward += reward

        # Visualization
        plt.clf()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().add_patch(plt.Rectangle((env.road_center - env.road_width / 2, 0), env.road_width, 1, color='gray'))
        plt.gca().add_patch(plt.Rectangle((observation[0] - 0.025, observation[1] - 0.025), 0.05, 0.05, color='blue'))
        plt.pause(0.01)

        # Capture the current figure as an image and save it
        frame_count += 1
        frame_filename = os.path.join(episode_directory, f'frame_{frame_count:04d}.png')

        plt.savefig(frame_filename)

        if done:
            break

# %%
import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt


class SimpleDrivingEnv(gym.Env):
    def __init__(self):
        super(SimpleDrivingEnv, self).__init__()

        # Define action space (3 actions: left, straight, right)
        self.action_space = spaces.Discrete(3)

        # Define observation space (2D position of the car)
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        # Initialize the car's position
        self.car_position = np.array([0.5, 0.1])

        # Define road parameters (in this example, a simple straight road)
        self.road_width = 0.2
        self.road_center = 0.5

    def reset(self):
        # Reset the car's position to the starting point
        self.car_position = np.array([0.5, 0.1])
        return self.car_position

    def step(self, action):
        # Define how the car moves based on the selected action
        if action == 0:  # Left
            self.car_position[0] -= 0.05
        elif action == 1:  # Straight (center)
            pass  # Car stays in the current lane
        elif action == 2:  # Right
            self.car_position[0] += 0.05

        # Simulate the car's movement
        self.car_position[1] += 0.05  # Car moves forward

        # Calculate reward based on the car's position
        reward = self.calculate_reward()

        # Check if the car has reached the end of the road
        done = self.car_position[1] >= 1.0

        # Return the current observation, reward, whether it's done, and additional info
        return self.car_position, reward, done, {}

    def calculate_reward(self):
        # Calculate reward based on the car's position relative to the road center
        distance_to_center = abs(self.car_position[0] - self.road_center)
        if distance_to_center < self.road_width / 2:
            return 1.0  # Maximum reward when the car is in the center of the road
        else:
            return 0.0  # No reward when the car is outside the road

    def render(self, mode='human'):
        if mode == 'human':
            plt.clf()
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.gca().add_patch(plt.Rectangle((self.road_center - self.road_width / 2, 0), self.road_width, 1, color='gray'))
            plt.gca().add_patch(plt.Rectangle((self.car_position[0] - 0.025, self.car_position[1] - 0.025), 0.05, 0.05, color='blue'))
            plt.pause(0.01)
        elif mode == 'rgb_array':
            raise NotImplementedError("Rendering as RGB array is not supported in this example.")
        

# Directory to save visualized images
image_directory = 'C:/Users/User/Documents/SELF DRIVING CAR/New folder'

# Create the environment
env = SimpleDrivingEnv()

# Reset the environment to get the initial observation
observation = env.reset()

# Visualization
plt.clf()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().add_patch(plt.Rectangle((env.road_center - env.road_width / 2, 0), env.road_width, 1, color='gray'))
plt.gca().add_patch(plt.Rectangle((observation[0] - 0.025, observation[1] - 0.025), 0.05, 0.05, color='blue'))

# Save the initial image
frame_count = 0
frame_filename = os.path.join(image_directory, f'frame_{frame_count:04d}.png')
plt.savefig(frame_filename)

while True:
    action = env.action_space.sample()

    observation, reward, done, _ = env.step(action)

    # Visualization
    plt.clf()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().add_patch(plt.Rectangle((env.road_center - env.road_width / 2, 0), env.road_width, 1, color='gray'))
    plt.gca().add_patch(plt.Rectangle((observation[0] - 0.025, observation[1] - 0.025), 0.05, 0.05, color='blue'))

    # Save the image
    frame_count += 1
    frame_filename = os.path.join(image_directory, f'frame_{frame_count:04d}.png')
    plt.savefig(frame_filename)

    if done:
        break

# Close the figure
plt.close()

print(f"Images saved in {image_directory}")   

# %%
import cv2
import os

# Directory containing the images
image_folder = 'C:/Users/User/Documents/SELF DRIVING CAR/car'

# Video file name and path
video_name = 'CAR_LEARNING.mp4'
output_directory = 'C:/Users/User/Documents/SELF DRIVING CAR/'

# Define the video path
video_path = os.path.join(output_directory, video_name)

# Define the desired width and height
width, height = 640, 480

# Get the list of image files and sort them numerically
images = [f"{i}.png" for i in range(1, 27)]  # Include all 26 images

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for mp4 format
video = cv2.VideoWriter(video_path, fourcc, 1, (width, height))

# Read and resize each image, then write it to the video
for image in images:
    img = cv2.imread(os.path.join(image_folder, image))
    img = cv2.resize(img, (width, height))
    video.write(img)

# Release the video writer and close any open windows
cv2.destroyAllWindows()
video.release()

print(f"Video saved to {video_path}")


# %%
import matplotlib.pyplot as plt
import numpy as np

# Mean Squared Error (MSE) values for Part 1 and Part 2
mse_values_part1 = [0.010407698542310793]
mse_values_part2 = [0.017402137033778194]

# Create labels for the x-axis
datasets = ['My dataset', 'Another dataset']

# Create x-values for the data points
x_values = np.arange(len(datasets))  # Use np.arange for more control over bar positions

# Define custom colors for the bars
colors = ['blue', 'red']

# Set custom bar width
bar_width = 0.35  

# Create a bar chart to visualize the comparison
plt.figure(figsize=(8, 6))
bars = plt.bar(x_values, mse_values_part1 + mse_values_part2, width=bar_width, color=colors)

plt.xlabel('Dataset')
plt.ylabel('MSE Value')
plt.title('Comparison of Mean Squared Error (MSE)')

# Set x-axis ticks and labels
plt.xticks(x_values, datasets)

# Show the plot
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np

# Error values for My Dataset and Another Dataset
error_values_my = [0.010407698542310793, 0.10201812849837422, 0.07333211928088365]
error_values_another = [0.017402137033778194, 0.13191715973965704, 0.09614446253885455]

# Create labels for the x-axis
datasets = ['MSE', 'RMSE', 'MAE']

# Create x-values for the data points
x_values = np.arange(len(datasets))

# Set custom bar width
bar_width = 0.3

# Create a bar chart to visualize the comparison
plt.figure(figsize=(10, 6))
plt.bar(x_values - bar_width, error_values_my, width=bar_width, label='My Dataset', color='Pink')
plt.bar(x_values, error_values_another, width=bar_width, label='Another Dataset', color='blue')

# Add labels, title, and legend
plt.xlabel('Error Metrics')
plt.ylabel('Error Values')
plt.title('Comparison of Error Metrics (MSE, RMSE, MAE) between My Dataset and Another Dataset')
plt.xticks(x_values - bar_width/2, datasets)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Error values for My Dataset and Another Dataset
error_values_my = [0.010407698542310793, 0.10201812849837422, 0.07333211928088365]
error_values_another = [0.017402137033778194, 0.13191715973965704, 0.09614446253885455]

# Create labels for the x-axis
datasets = ['MSE', 'RMSE', 'MAE']

# Create x-values for the data points
x_values = np.arange(len(datasets))

# Set custom bar width
bar_width = 0.3

# Create a 3D figure
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Position of the bars on the x-axis
x_position_my = x_values - bar_width/2
x_position_another = x_values + bar_width/2

# Create the bars for My Dataset and Another Dataset
ax.bar(x_position_my, error_values_my, zs=0, zdir='y', width=bar_width, label='My Dataset', color='pink')
ax.bar(x_position_another, error_values_another, zs=1, zdir='y', width=bar_width, label='Another Dataset', color='blue')

# Add labels, title, and legend
ax.set_xlabel('Error Metrics')
ax.set_ylabel('Dataset')
ax.set_zlabel('Error Values')
ax.set_title('3D Comparison of Error Metrics (MSE, RMSE, MAE) between My Dataset and Another Dataset')
ax.set_yticks([0, 1])
ax.set_yticklabels(['My Dataset', 'Another Dataset'])
ax.legend()

# Show the plot
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Error values for My Dataset and Another Dataset
error_values_my = [0.010407698542310793, 0.10201812849837422, 0.07333211928088365]
error_values_another = [0.017402137033778194, 0.13191715973965704, 0.09614446253885455]

# Create labels for the x-axis
datasets = ['MSE', 'RMSE', 'MAE']

# Create x-values for the data points
x_values = np.arange(len(datasets))

# Set custom bar width
bar_width = 0.3

# Create a 3D figure
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Position of the bars on the x-axis
x_position_my = x_values - bar_width/2
x_position_another = x_values + bar_width/2

# Create the bars for My Dataset and Another Dataset
ax.bar(x_position_my, error_values_my, zs=0, zdir='y', width=bar_width, label='My Dataset', color='pink')
ax.bar(x_position_another, error_values_another, zs=1, zdir='y', width=bar_width, label='Another Dataset', color='blue')

# Add labels, title, and legend
ax.set_xlabel('Error Metrics')
ax.set_ylabel('Dataset')
ax.set_zlabel('Error Values')
ax.set_title('3D Comparison of Error Metrics (MSE, RMSE, MAE) between My Dataset and Another Dataset')
ax.set_yticks([0, 1])
ax.set_yticklabels(['My Dataset', 'Another Dataset'])
ax.legend()

# Adjust the layout and spacing manually
ax.dist = 12  # Adjust the distance to control the perspective
ax.view_init(elev=20, azim=-45)  # Adjust the view angle

# Show the plot
plt.show()

# Create a DataFrame from the error values
data = {
    'Error Metrics': datasets,
    'My Dataset': error_values_my,
    'Another Dataset': error_values_another
}
df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
df.to_excel('error_metrics_data.xlsx', index=False)



