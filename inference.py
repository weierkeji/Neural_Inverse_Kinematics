import torch
from modules.modules import *
import ikpy.chain
import numpy as np
import argparse

def inference(cfg):
    r_arm = ikpy.chain.Chain.from_urdf_file(cfg.chain_path)
    upper = []
    lower = []
    sampled = []
    for i in range(1, len(r_arm.links) - 1):
        lower.append(r_arm.links[i].bounds[0])
        upper.append(r_arm.links[i].bounds[1])
    upper = np.array(upper)
    lower = np.array(lower)
 
    hypernet = HyperNet(cfg)
    mainnet = MainNet(cfg)
    hypernet.load_state_dict(torch.load('best_model.pt', map_location=torch.device('cpu')))  #CPU only
    # hypernet.load_state_dict(torch.load('best_model.pt'))  #GPU
    hypernet.eval()

    positions = torch.tensor([[0.1723, 0.007497, 0.49]], dtype=torch.float)  #TODO
    joint_angles = torch.tensor([[0,0,0,0,0,0]], dtype=torch.float)  # provide shape information
    predicted_weights = hypernet(positions)
    for j in range(cfg.num_solutions_validation):
        samples, distributions, means, variance, selection = mainnet.validate(torch.ones(joint_angles.shape[0], 1), predicted_weights, lower, upper)
        sampled.append(samples)

    for sampled_lst in sampled:
        for k in range(len(positions)):
            joint_angles = [0] + [sampled_lst[i][k].item() for i in range(cfg.num_joints)] + [0]
            print(joint_angles)
            real_frame = r_arm.forward_kinematics(joint_angles)
            print(real_frame[:3, 3])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain-path', type=str, default="assets/mycobot/new_mycobot_pro_320_pi_2022.urdf", help='urdf chain path')
    parser.add_argument('--num-joints', type=int, default=6, help='number of joints of the kinematic chain')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num-solutions-validation', type=int, default=10, help='learning rate')
    parser.add_argument('--grad-clip', type=int, default=1, help='clip norm of gradient')
    parser.add_argument('--embedding-dim', type=int, default=128, help='embedding dimension')
    parser.add_argument('--hypernet-input-dim', type=int, default=3, help='number of input to the hypernetwork (f). default case 3 (x, y, z)')
    parser.add_argument('--hypernet-hidden-size', type=int, default=1024, help='hypernetwork (f) number of neurons in hidden layer')
    parser.add_argument('--hypernet-num-hidden-layers', type=int, default=3, help='hypernetwork  (f) number of hidden layers')
    parser.add_argument('--jointnet-hidden-size', type=int, default=256, help='jointnet (g) number of neurons in hidden layer')
    parser.add_argument('--num-gaussians', type=int, default=50, help='number of gaussians for mixture . default=1 no mixture')

    parser.set_defaults()
    cfg = parser.parse_args()

    cfg.jointnet_output_dim = cfg.num_gaussians * 2 + cfg.num_gaussians if cfg.num_gaussians != 1 else 2

    inference(cfg)



