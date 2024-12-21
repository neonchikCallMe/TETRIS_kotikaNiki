import os
from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean
from logs import CustomTensorBoard
from tqdm import tqdm

def dqn():
    env = Tetris()
    episodes = 3000  
    max_steps = None  
    epsilon_stop_episode = 2000  
    mem_size = 1000 
    discount = 0.95  
    batch_size = 128  
    epochs = 1  
    render_every = 50  
    render_delay = None  
    log_every = 50  
    replay_start_size = 1000  
    train_every = 1  
    n_neurons = [32, 32, 32]  
    activations = ['relu', 'relu', 'relu', 'linear']  
    save_best_model = True  
    
    model_file = "best.keras"
    
    if os.path.exists(model_file):
        print(f"Loading existing model from {model_file}")
        agent = DQNAgent(env.get_state_size(),
                         n_neurons=n_neurons, activations=activations,
                         epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                         discount=discount, replay_start_size=replay_start_size,
                         modelFile=model_file)  
    else:
        print("No existing model found, starting from scratch.")
        agent = DQNAgent(env.get_state_size(),
                         n_neurons=n_neurons, activations=activations,
                         epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                         discount=discount, replay_start_size=replay_start_size)  
    
    log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    log = CustomTensorBoard(log_dir=log_dir)

    scores = []
    best_score = 0

    for episode in tqdm(range(episodes)):
        current_state = env.reset()
        done = False
        steps = 0

        if render_every and episode % render_every == 0:
            render = True
        else:
            render = False

        # Game
        while not done and (not max_steps or steps < max_steps):

            next_states = {tuple(v): k for k, v in env.get_next_states().items()}
            best_state = agent.best_state(next_states.keys())
            best_action = next_states[best_state]

            reward, done = env.play(best_action[0], best_action[1], render=render,
                                    render_delay=render_delay)

            agent.add_to_memory(current_state, best_state, reward, done)
            current_state = best_state
            steps += 1

        scores.append(env.get_game_score())

        if episode % train_every == 0:
            agent.train(batch_size=batch_size, epochs=epochs)

        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])

            log.log(episode, avg_score=avg_score, min_score=min_score,
                    max_score=max_score)

        if save_best_model and env.get_game_score() > best_score:
            print(f'Saving a new best model (score={env.get_game_score()}, episode={episode})')
            best_score = env.get_game_score()
            agent.save_model(model_file)

if __name__ == "__main__":
    dqn()
