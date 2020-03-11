import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from pathlib import Path
from dataset.dataset_class import FineTuningImagesDataset, FineTuningVideoDataset
from network.model import Generator, Discriminator
from loss.loss_discriminator import LossDSCreal, LossDSCfake
from loss.loss_generator import LossGF
from network.utils import calculate_fid
from config import (
    device,
    cpu,
    path_to_chkpt,
    path_to_video,
    path_to_images,
    path_to_save,
    path_to_e_hat_video,
    VGGFace_body_path,
    VGGFace_weight_path,
)


"""Create dataset and net"""
choice = ''
while choice != '0' and choice != '1':
    choice = input(
        'What source to finetune on?\n0: Video\n1: Images\n\nEnter number\n>>'
    )
if choice == '0':  # video
    dataset = FineTuningVideoDataset(path_to_video, device)
else:  # Images
    dataset = FineTuningImagesDataset(path_to_images, device)
dataLoader = DataLoader(dataset, batch_size=2, shuffle=False)

# Initialize SummaryWriter for tensorboard
RUN_NAME = datetime.now().strftime(format='%b%d_%H-%M-%S')
writer = SummaryWriter(log_dir=f'runs/{RUN_NAME}')

# Initialize PartialInceptionNetwork for calculating FID score
inception = PartialInceptionNetwork().eval()
# Initialize Cosine similarity for calculating CSIM score
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

e_hat = torch.load(path_to_e_hat_video, map_location=cpu)
e_hat = e_hat['e_hat']

G = Generator(256, finetuning=True, e_finetuning=e_hat)
D = Discriminator(dataset.__len__(), finetuning=True, e_finetuning=e_hat)

G.train()
D.train()

optimizerG = optim.Adam(params=G.parameters(), lr=5e-5)
optimizerD = optim.Adam(params=D.parameters(), lr=2e-4)


"""Criterion"""
criterionG = LossGF(
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

num_epochs = 40

# Warning if checkpoint inexistant
if not Path(path_to_chkpt).is_file():
    print('ERROR: cannot find checkpoint')


"""Loading from past checkpoint"""
checkpoint = torch.load(path_to_chkpt, map_location=cpu)
checkpoint['D_state_dict']['W_i'] = torch.rand(
    512, dataset.__len__()
)  # change W_i for finetuning

G.load_state_dict(checkpoint['G_state_dict'])
D.load_state_dict(checkpoint['D_state_dict'])


"""Change to finetuning mode"""
G.finetuning_init()
D.finetuning_init()

G.to(device)
D.to(device)

"""Training"""
batch_start = datetime.now()

for epoch in range(num_epochs):
    for i_batch, (x, g_y) in enumerate(dataLoader):
        with torch.autograd.enable_grad():
            # zero the parameter gradients
            optimizerG.zero_grad()
            optimizerD.zero_grad()

            # forward
            # train G and D
            x_hat = G(g_y, e_hat)
            r_hat, D_hat_res_list = D(x_hat, g_y, i=0)
            r, D_res_list = D(x, g_y, i=0)

            lossG = criterionG(x, x_hat, r_hat, D_res_list, D_hat_res_list)
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
            r_hat, D_hat_res_list = D(x_hat, g_y, i=0)
            r, D_res_list = D(x, g_y, i=0)

            lossDfake = criterionDfake(r_hat)
            lossDreal = criterionDreal(r)

            lossD = lossDreal + lossDfake
            lossD.backward(retain_graph=False)
            optimizerD.step()

        # Output training stats
        if epoch % 10 == 0:
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

        lossesD.append(lossD.item())
        lossesG.append(lossG.item())

writer.close()
plt.clf()
plt.plot(lossesG)  # blue
plt.plot(lossesD)  # orange
plt.show()

print('Saving finetuned model...')
torch.save(
    {
        'epoch': epoch,
        'lossesG': lossesG,
        'lossesD': lossesD,
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
    },
    path_to_save,
)
print('...Done saving latest')
