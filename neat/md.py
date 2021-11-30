import gym
import numpy as np
import cv2 
import neat
import pickle

from gym.wrappers import ResizeObservation

env = gym.make('RetroArch-v0')
env = ResizeObservation(env, 240)
imgarray = []


def eval_genomes(genomes, config):


    for genome_id, genome in genomes:
        ob = env.reset()
        ac = env.action_space.sample()

        inx, iny, inc = env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        
        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 200
        xpos = 0
        xpos_max = 0
        
        done = False

        while not done:
            
            env.render()
            frame += 1
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx,iny))
            
            imgarray = np.ndarray.flatten(ob)
            nnOutput = net.activate(imgarray)
            #nn_act = nnOutput[0:18]
            max_index = np.argmax([nnOutput])
            ob, rew, done, info = env.step(max_index)
                        
            fitness_current += rew

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter += 100
            else:
                counter -= 1
                
            if done or counter == 0:
                done = True
                print(genome_id, fitness_current)
                
            genome.fitness = fitness_current
    

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config)
#p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-126')

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
