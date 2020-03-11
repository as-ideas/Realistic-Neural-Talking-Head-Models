import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from matplotlib import pyplot as plt
import pytorch_ssim
import os
import numpy as np

from dataset.dataset_class import VidDataSet
from loss.loss_discriminator import LossDSCreal, LossDSCfake
from loss.loss_generator import LossG
from network.model import Generator, Embedder, Discriminator, PartialInceptionNetwork
from network.utils import calculate_fid
from config import (
    device,
    cpu,
    path_to_chkpt,
    path_to_backup,
    num_epochs,
    VGGFace_body_path,
    VGGFace_weight_path,
)

"""Create dataset and net"""
dataset = VidDataSet(K=8, path_to_mp4='mp4', device=device)
dataLoader = DataLoader(dataset, batch_size=2, shuffle=True)
print(dataLoader)
# Initialize SummaryWriter for tensorboard
RUN_NAME = datetime.now().strftime(format='%b%d_%H-%M-%S')
writer = SummaryWriter(log_dir=f'runs/{RUN_NAME}')

# Initialize PartialInceptionNetwork for calculating FID score
inception = PartialInceptionNetwork().eval()
# Initialize Cosine similarity for calculating CSIM score
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

G = Generator(224).to(device)
E = Embedder(224).to(device)
D = Discriminator(dataset.__len__()).to(device)

G.train()
E.train()
D.train()

optimizerG = optim.Adam(params=list(E.parameters()) + list(G.parameters()), lr=5e-5)
optimizerD = optim.Adam(params=D.parameters(), lr=2e-4)


"""Criterion"""
criterionG = LossG(
    VGGFace_body_path=VGGFace_body_path,
    VGGFace_weight_path=VGGFace_weight_path,
    device=device,
)
criterionDreal = LossDSCreal()
criterionDfake = LossDSCfake()


"""Training init"""
epochCurrent = epoch = i_batch = 0
lossesG = []
lossesD = []
i_batch_current = 0


# initiate checkpoint if inexistant
if not os.path.isfile(path_to_chkpt):
    print('Initiating new checkpoint...')
    torch.save(
        {
            'epoch': epoch,
            'lossesG': lossesG,
            'lossesD': lossesD,
            'E_state_dict': E.state_dict(),
            'G_state_dict': G.state_dict(),
            'D_state_dict': D.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            'num_vid': dataset.__len__(),
            'i_batch': i_batch,
        },
        path_to_chkpt,
    )
    print('...Done')

"""Loading from past checkpoint"""
checkpoint = torch.load(path_to_chkpt, map_location=cpu)
E.load_state_dict(checkpoint['E_state_dict'])
G.load_state_dict(checkpoint['G_state_dict'])
D.load_state_dict(checkpoint['D_state_dict'])
optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
epochCurrent = checkpoint['epoch']
lossesG = checkpoint['lossesG']
lossesD = checkpoint['lossesD']
num_vid = checkpoint['num_vid']
i_batch_current = checkpoint['i_batch'] + 1

G.train()
E.train()
D.train()


"""Training"""
batch_start = datetime.now()

for epoch in range(epochCurrent, num_epochs):
    for i_batch, (f_lm, x, g_y, i) in enumerate(dataLoader, start=i_batch_current):
        if i_batch > len(dataLoader):
            i_batch_current = 0
            break
        with torch.autograd.enable_grad():
            # zero the parameter gradients
            optimizerG.zero_grad()
            optimizerD.zero_grad()

            # forward
            # Calculate average encoding vector for video
            f_lm_compact = f_lm.view(
                -1, f_lm.shape[-4], f_lm.shape[-3], f_lm.shape[-2], f_lm.shape[-1]
            )  # BxK,2,3,224,224

            e_vectors = E(
                f_lm_compact[:, 0, :, :, :], f_lm_compact[:, 1, :, :, :]
            )  # BxK,512,1
            e_vectors = e_vectors.view(-1, f_lm.shape[1], 512, 1)  # B,K,512,1
            e_hat = e_vectors.mean(dim=1)

            # train G and D
            x_hat = G(g_y, e_hat)
            r_hat, D_hat_res_list = D(x_hat, g_y, i)
            r, D_res_list = D(x, g_y, i)

            lossG = criterionG(
                x, x_hat, r_hat, D_res_list, D_hat_res_list, e_vectors, D.W_i, i
            )
            lossDfake = criterionDfake(r_hat)
            lossDreal = criterionDreal(r)

            loss = lossDreal + lossDfake + lossG
            loss.backward(retain_graph=False)
            optimizerG.step()
            optimizerD.step()

            # train D again
            optimizerG.zero_grad()
            optimizerD.zero_grad()
            x_hat.detach_()
            r_hat, D_hat_res_list = D(x_hat, g_y, i)
            r, D_res_list = D(x, g_y, i)

            lossDfake = criterionDfake(r_hat)
            lossDreal = criterionDreal(r)

            lossD = lossDreal + lossDfake
            lossD.backward(retain_graph=False)
            optimizerD.step()

        # Output training stats
        if i_batch % 10 == 0:
            batch_end = datetime.now()
            avg_time = (batch_end - batch_start) / 10
            print('\n\navg batch time for batch size of', x.shape[0], ':', avg_time)

            batch_start = datetime.now()

            print(
                '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(y)): %.4f'
                % (
                    epoch,
                    num_epochs,
                    i_batch,
                    len(dataLoader),
                    lossD.item(),
                    lossG.item(),
                    r.mean(),
                    r_hat.mean(),
                )
            )

            # Initialize empty lists for storing metrics
            ssim_vals = []
            csim_vals = []
            fid_vals = []

            # Calculate the metrics
            fid = calculate_fid(x, x_hat, inception)
            fid_vals.append(fid)

            for img_no in range(x_hat.shape[0]):
                x_temp = x.transpose(1, 3)[img_no].unsqueeze(0)
                x_hat_temp = x_hat.transpose(1, 3)[img_no].unsqueeze(0)

                # calcule the batch avg SSIM score
                ssim_val = pytorch_ssim.ssim(x_temp, x_hat_temp)
                ssim_vals.append(ssim_val)

                # calcule the batch avg CSIM score
                csim_val = cos(x_temp, x_hat_temp)
                csim_vals.append(csim_val)

            # Calculate the mean of metrics stored in each list
            ssim_score = torch.mean(torch.stack(ssim_vals))
            csim_score = torch.cat((csim_vals[0], csim_vals[1]), dim=1)
            fid_score = torch.mean(torch.stack(fid_vals))

            # Log metrics and losses in tensorboard
            writer.add_scalar('SSIM', ssim_score, i_batch)
            writer.add_scalar('CSIM', ssim_score, i_batch)
            writer.add_scalar('FID', ssim_score, i_batch)
            writer.add_scalar('LossD', lossD.item(), i_batch)
            writer.add_scalar('LossG', lossG.item(), i_batch)

            plt.clf()
            out = x_hat.transpose(1, 3)[0]
            for img_no in range(1, x_hat.shape[0]):
                out = torch.cat((out, x_hat.transpose(1, 3)[img_no]), dim=1)
            out = out.type(torch.int32).to(cpu).numpy()
            writer.add_image(
                'test_x_hat_image',
                np.transpose(out.astype("uint8"), (2, 0, 1)),
                i_batch,
            )
            plt.imshow(out)
            plt.show()

            plt.clf()
            out = x.transpose(1, 3)[0]
            for img_no in range(1, x.shape[0]):
                out = torch.cat((out, x.transpose(1, 3)[img_no]), dim=1)
            out = out.type(torch.int32).to(cpu).numpy()
            writer.add_image(
                'test_x_image', np.transpose(out.astype("uint8"), (2, 0, 1)), i_batch
            )
            plt.imshow(out)
            plt.show()

            plt.clf()
            out = g_y.transpose(1, 3)[0]
            for img_no in range(1, g_y.shape[0]):
                out = torch.cat((out, g_y.transpose(1, 3)[img_no]), dim=1)
            out = out.type(torch.int32).to(cpu).numpy()
            writer.add_image(
                'test_g_y_image', np.transpose(out.astype("uint8"), (2, 0, 1)), i_batch
            )
            plt.imshow(out)
            plt.show()

        if i_batch % 100 == 99:
            lossesD.append(lossD.item())
            lossesG.append(lossG.item())

            plt.clf()
            plt.plot(lossesG)  # blue
            plt.plot(lossesD)  # orange
            plt.show()

            print('Saving latest...')
            torch.save(
                {
                    'epoch': epoch,
                    'lossesG': lossesG,
                    'lossesD': lossesD,
                    'E_state_dict': E.state_dict(),
                    'G_state_dict': G.state_dict(),
                    'D_state_dict': D.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    'num_vid': dataset.__len__(),
                    'i_batch': i_batch,
                },
                path_to_chkpt,
            )
            print('...Done saving latest')

        if i_batch % 500 == 499:

            print('Saving latest...')
            torch.save(
                {
                    'epoch': epoch,
                    'lossesG': lossesG,
                    'lossesD': lossesD,
                    'E_state_dict': E.state_dict(),
                    'G_state_dict': G.state_dict(),
                    'D_state_dict': D.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    'num_vid': dataset.__len__(),
                    'i_batch': i_batch,
                },
                path_to_backup,
            )
            print('...Done saving latest')

writer.close()
