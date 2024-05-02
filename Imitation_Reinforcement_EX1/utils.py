import numpy as np

def clean_labels(y_data):
    """ 
    remove labels with multiple actions
    """
    print("original labels:", len(y_data))
    # remove labels with multiple actions, only one index should be not zero
    y_data_new = []
    for i in range(len(y_data)):
        if sum(abs(y_data[i])) == 1:
            y_data_new.append(y_data[i])
        elif sum(abs(y_data[i])) == 0:
            y_data_new.append(y_data[i])
        elif np.isclose(0.2, sum(abs(y_data[i]))):
            y_data_new.append(y_data[i])

    print("cleaned labels:", len(y_data_new))
    return np.array(y_data_new)


def upsample(X_train, y_train):
    """
    Upsample the data so that the classes are balanced
    """

    # Count class occurrences
    unique, counts = np.unique(y_train, axis=0, return_counts=True)
    
    count_cls = len(y_train) // len(unique)
    
    # for each class, upsample to count_cls, or downsample to count_cls
    X_train_new = []
    y_train_new = []
    for i in range(len(unique)):
        indices = np.where((y_train == unique[i]).all(axis=1))[0]
        if len(indices) > count_cls:
            indices = np.random.choice(indices, count_cls, replace=False)
        else:
            indices = np.random.choice(indices, count_cls, replace=True)
        X_train_new.append(X_train[indices])
        y_train_new.append(y_train[indices])
        
    X_train_new = np.concatenate(X_train_new)
    y_train_new = np.concatenate(y_train_new)
    
    unique, counts = np.unique(y_train_new, axis=0, return_counts=True)

    return X_train_new, y_train_new

def convert_labels(y_data):
    """
    Converts actions to labels
    """

    # Define mapping between old and new labels
    mapping = {
        (-1, 0, 0): [1, 0, 0, 0, 0], # left
        (1, 0, 0):  [0, 1, 0, 0, 0], # right
        (0, 1, 0):  [0, 0, 1, 0, 0], # acc
        (0, 0, 0.2):[0, 0, 0, 1, 0], # break
        (0, 0, 0):  [0, 0, 0, 0, 1],  # straight
    }

    # Convert labels using indexing
    new_labels = np.zeros((len(y_data), 5)) 
    for old_label, new_label in mapping.items():
        matches = (y_data == old_label).all(axis=1)
        new_labels[matches] = new_label

    return new_labels

def convert_actions(out, max_speed=1.0):
    """
    Converts label probs to action, for imatation learning task
    """
    left = 0 # -> [-1, 0, 0]
    right = 1 # -> [1, 0, 0]
    acc = 2 # -> [0, 1, 0]
    brake = 3 # -> [0, 0, 0.2]
    straight = 4 # -> [0, 0, 0]

    a = np.argmax(out)
    
    # Define mapping between old and new labels
    mapping = {
        left: [-1, 0, 0.05],
        right: [1, 0, 0.05],
        acc: [0, max_speed, 0],
        brake: [0, 0, 0.2],
        straight: [0, 0, 0],
    }
    
    return mapping[a]

def rgb2gray(rgb):
    """
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[..., :3], [0.2125, 0.7154, 0.0721])
    if len(gray.shape) == 3:
        gray= gray[:, :84, 6:90]
    else:
        gray = gray[:84, 6:90]
    return gray.astype("float32")


def action_to_id(a):
    """
    this method discretizes the actions.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    if all(a == [-1.0, 0.0, 0.0]):
        return LEFT  # LEFT: 1
    elif all(a == [1.0, 0.0, 0.0]):
        return RIGHT  # RIGHT: 2
    elif all(a == [0.0, 1.0, 0.0]):
        return ACCELERATE  # ACCELERATE: 3
    elif all(a == [0.0, 0.0, 0.2]):
        return BRAKE  # BRAKE: 4
    else:
        return STRAIGHT  # STRAIGHT = 0


def add_history(X, Y, history_length=1):
    """
    this method changes the shape of the data to include history.
    """
    X_new = []
    Y_new = []
    history_length += 1
    # starting from the history_length-th element, append the last history_length elements to the current element
    for i in range(history_length, len(X)):
        x = X[i - history_length : i]
        y = Y[i]
        X_new.append(x)
        Y_new.append(y)
    return np.array(X_new), np.array(Y_new)
    
def id_to_action(action_id, max_speed=1.0):
    """
    this method makes actions continous.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    
    LEFT = 0
    RIGHT = 1
    ACCELERATE = 2
    BRAKE = 3
    STRAIGHT = 4

    a = np.array([0.0, 0.0, 0.0])

    if action_id == LEFT:
        return np.array([-1.0, 0.0, 0.05])
    elif action_id == RIGHT:
        return np.array([1.0, 0.0, 0.05])
    elif action_id == ACCELERATE:
        return np.array([0.0, max_speed, 0.0])
    elif action_id == BRAKE:
        return np.array([0.0, 0.0, 0.1])
    else:
        return np.array([0.0, 0.0, 0.0])


class EpisodeStats:
    """
    This class tracks statistics like episode reward or action usage.
    """

    def __init__(self):
        self.episode_reward = 0
        self.actions_ids = []

    def step(self, reward, action_id):
        self.episode_reward += reward
        self.actions_ids.append(action_id)

    def get_action_usage(self, action_id):
        ids = np.array(self.actions_ids)
        return len(ids[ids == action_id]) / len(ids)
