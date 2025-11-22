import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from controller import Supervisor  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# red neuronal DQN
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # parametros
        self.gamma = 0.99 
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.target_update = 100
        
        # redes
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        self.memory = ReplayBuffer(10000)
        
        self.steps_done = 0
        self.last_distance = None
    
    def select_action(self, state):
        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        loss = self.criterion(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class WebotsEnv:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.timestep = int(supervisor.getBasicTimeStep())
        
        # nodo del robot
        self.robot_node = supervisor.getFromDef("ROBOT")
        if self.robot_node is None:
            print("ERROR: No se encontró el robot con DEF 'ROBOT'")
            print("Asegúrate de que tu robot tenga DEF 'ROBOT' en el archivo .wbt")
            self.robot_node = supervisor.getSelf()
            if self.robot_node is None:
                raise Exception("No se pudo obtener el nodo del robot")
            else:
                print("Usando getSelf() como alternativa")
        
        self.initial_translation = self.robot_node.getField("translation").getSFVec3f()
        self.initial_rotation = self.robot_node.getField("rotation").getSFRotation()
        
        self.lidar = supervisor.getDevice('lidar')
        self.lidar.enable(self.timestep)
        
        self.gps = supervisor.getDevice('gps')
        self.gps.enable(self.timestep)
        
        self.compass = supervisor.getDevice('compass')
        self.compass.enable(self.timestep)
        
        self.left_motor = supervisor.getDevice('left wheel')
        self.right_motor = supervisor.getDevice('right wheel')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        self.goal_position = np.array([1.4, 1.3])  # [X, Y]
        
        self.max_speed = 4.0 
        self.action_space = 4
        self.previous_distance = None
        self.steps_without_progress = 0
        
        # inicialización
        for _ in range(10):
            supervisor.step(self.timestep)
    
    def get_state(self):
        ranges = self.lidar.getRangeImage()
        if not ranges:
            lidar_data = np.zeros(8)
        else:
            num_sectors = 8
            sector_size = len(ranges) // num_sectors
            lidar_data = []
            for i in range(num_sectors):
                start = i * sector_size
                end = start + sector_size
                min_dist = min(ranges[start:end])
                lidar_data.append(min(min_dist, 5.0) / 5.0)  
            lidar_data = np.array(lidar_data)
        
        pos = self.gps.getValues()
        current_pos = np.array([pos[0], pos[1]])  
        
        # distancia y ángulo a la meta
        diff = self.goal_position - current_pos
        distance = np.linalg.norm(diff)
        angle = np.arctan2(diff[1], diff[0])
        
        north = self.compass.getValues()
        heading = np.arctan2(north[0], north[1])
        
        relative_angle = angle - heading
        relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))
        
        state = np.concatenate([
            lidar_data,
            [distance / 10.0],  
            [relative_angle / np.pi] 
        ])
        
        return state, distance, current_pos
    
    def step(self, action):
        # acciones
        if action == 0:  # adelante
            self.left_motor.setVelocity(self.max_speed)
            self.right_motor.setVelocity(self.max_speed)
        elif action == 1:  # izquierda
            self.left_motor.setVelocity(self.max_speed * 0.3)
            self.right_motor.setVelocity(self.max_speed)
        elif action == 2:  # derecha
            self.left_motor.setVelocity(self.max_speed)
            self.right_motor.setVelocity(self.max_speed * 0.3)
        elif action == 3:  # atrás
            self.left_motor.setVelocity(-self.max_speed * 0.5)
            self.right_motor.setVelocity(-self.max_speed * 0.5)
        
        self.supervisor.step(self.timestep)
        
        next_state, distance, current_pos = self.get_state()
        reward = 0
        done = False
        
        # recompensa por avanzar hacia la meta
        if self.previous_distance is not None:
            progress = self.previous_distance - distance
            reward += progress * 20 
            
            if abs(progress) < 0.005: 
                self.steps_without_progress += 1
            else:
                self.steps_without_progress = 0
        
        self.previous_distance = distance
        
        # recompensa grande por llegar a la meta
        if distance < 0.5:
            reward += 200
            done = True
            print(f"¡META ALCANZADA! Distancia final: {distance:.3f}m")
        
        # penalización por colisión
        ranges = self.lidar.getRangeImage()
        if ranges:
            min_range = min(ranges)
            if min_range < 0.25:
                reward -= 50
                done = True
                print("Colisión detectada")
        
        # penalización por estancamiento
        if self.steps_without_progress > 150:  
            reward -= 20
            done = True
            print("Robot estancado")
        
        # penalización por tiempo 
        reward -= 0.05
        
        return next_state, reward, done
    
    def reset(self):
        """Resetear el robot a su posición inicial"""
        print("Reseteando robot a posición inicial...")
        
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        
        # Resetear posición, rotación y velocidad
        translation_field = self.robot_node.getField("translation")
        rotation_field = self.robot_node.getField("rotation")
        translation_field.setSFVec3f(self.initial_translation)
        rotation_field.setSFRotation(self.initial_rotation)
        self.robot_node.resetPhysics()
        
        for _ in range(10):
            self.supervisor.step(self.timestep)
        
        self.previous_distance = None
        self.steps_without_progress = 0
        
        state, _, _ = self.get_state()
        return state

def main():
    supervisor = Supervisor()
    env = WebotsEnv(supervisor)
    
    state_size = 10  # 8 LIDAR + distancia + ángulo
    action_size = 4
    
    agent = DQNAgent(state_size, action_size)
    
    # Entrenamiento
    num_episodes = 200
    max_steps = 1000
    episode_rewards = []
    success_count = 0
    
    print("=" * 60)
    print("Iniciando entrenamiento DQN con CUDA")
    print(f"Posición de la meta: {env.goal_position}")
    print(f"Posición inicial: {env.initial_translation[:2]}")
    print("=" * 60)
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        reached_goal = False
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            
            # Entrenar
            loss = agent.train()
            
            episode_reward += reward
            state = next_state
            agent.steps_done += 1
            
            if agent.steps_done % agent.target_update == 0:
                agent.update_target_network()
                print(f"Red objetivo actualizada en step {agent.steps_done}")
            
            if done:
                if reward > 100:  # Llegó a la meta
                    reached_goal = True
                    success_count += 1
                break
            
            if supervisor.step(0) == -1:
                print("Simulación terminada")
                return
        
        agent.decay_epsilon()
        
        episode_rewards.append(episode_reward)
        
        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
        success_rate = (success_count / (episode + 1)) * 100
        
        print(f"Episodio {episode+1}/{num_episodes} | "
              f"Reward: {episode_reward:.2f} | "
              f"Avg(10): {avg_reward:.2f} | "
              f"Steps: {step+1} | "
              f"Epsilon: {agent.epsilon:.3f} | "
              f"Success: {success_count} ({success_rate:.1f}%) | "
              f"Meta: {'✓' if reached_goal else '✗'}")
        
        if (episode + 1) % 20 == 0:
            torch.save({
                'episode': episode,
                'model_state_dict': agent.policy_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'rewards': episode_rewards
            }, f'dqn_checkpoint_ep{episode+1}.pth')
    
    torch.save({
        'episode': num_episodes,
        'model_state_dict': agent.policy_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'rewards': episode_rewards
    }, 'dqn_model_final.pth')
    

if __name__ == "__main__":
    main()