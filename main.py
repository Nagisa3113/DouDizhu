import os
import argparse
import torch
import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import get_device, set_seed, tournament, reorganize, Logger, plot_curve
from datetime import datetime

from agent.mcnfsp import MCNFSPAgent
from util.loadfile import loadconfig
from agent.nfsp import NFSPAgent


def train(args):
    # Check whether gpu is available
    device = get_device()
    # Seed numpy, torch, random
    set_seed(args.seed)
    env = rlcard.make(args.env, config={'seed': args.seed})

    paras = loadconfig(args)

    agents = []
    for index in range(env.num_players):
        agent = MCNFSPAgent(args=paras,
                            num_actions=env.num_actions,
                            state_shape=env.state_shape[index],
                            device=device, )
        agents.append(agent)

    # a1=NFSPAgent(args=paras,
    #                       num_actions=env.num_actions,
    #                       state_shape=env.state_shape[0],
    #                       device=device, )
    # a2=NFSPAgent(args=paras,
    #                       num_actions=env.num_actions,
    #                       state_shape=env.state_shape[1],
    #                       device=device, )
    # agents.append(a1)
    # agents.append(a2)
    # agents.append(a2)
    env.set_agents(agents)

    eval_env = rlcard.make(args.env, config={'seed': args.seed})
    eval_env.set_agents([agents[0], RandomAgent(num_actions=env.num_actions), RandomAgent(num_actions=env.num_actions)])

    # Start training
    train_para = paras.getpara('trainparameters')
    with Logger(args.log_dir) as logger:
        paras.save_config(args.algo, args.log_dir)

        for episode in range(train_para.num_episodes):
            trajectories, payoffs = env.run(is_training=True)
            trajectories = reorganize(trajectories, payoffs)
            for index in range(env.num_players):
                for ts in trajectories[index]:
                    agents[index].feed(ts)
            if episode > 0 and episode % train_para.eval_every == 0:
                print('\nepisode: ' + str(episode))
                logger.log_performance(env.timestep, tournament(eval_env, train_para.num_eval_games)[0])
        csv_path, fig_path = logger.csv_path, logger.fig_path
    plot_curve(csv_path, fig_path, args.algo)

    for index in range(env.num_players):
        save_path = os.path.join(args.log_dir, 'model' + str(index) + '.pth')
        torch.save(agents[index], save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("NFSP in RLCard")
    parser.add_argument('--env', type=str, default='doudizhu')
    parser.add_argument('--algo', type=str, default='nfsp')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_dir', type=str, default='results/bestmc')
    parser.add_argument('--config_dir', type=str, default='config/nfsp.yaml')

    args = parser.parse_args()

    startTime = datetime.now()
    print("Start Time =", startTime)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)

    endTime = datetime.now()
    trainingTime = endTime - startTime;
    print("Start Time: ", startTime.strftime("%m/%d-%H:%M"))
    print("End Time: ", endTime.strftime("%m/%d-%H:%M"))
    print("Training Time: ", trainingTime)
