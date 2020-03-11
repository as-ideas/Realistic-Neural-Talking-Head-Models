import torch
import cv2
from matplotlib import pyplot as plt

from network.model import Generator
from webcam_demo.webcam_extraction_conversion import generate_landmarks
from config import device, cpu, path_to_e_hat_video, path_to_chkpt


checkpoint = torch.load(path_to_chkpt, map_location=cpu)
e_hat = torch.load(path_to_e_hat_video, map_location=cpu)
e_hat = e_hat['e_hat'].to(device)

G = Generator(256, finetuning=True, e_finetuning=e_hat)
G.eval()

"""Training Init"""
G.load_state_dict(checkpoint['G_state_dict'])
G.to(device)
G.finetuning_init()

print('PRESS Q TO EXIT')
cap = cv2.VideoCapture(0)

with torch.no_grad():
    while True:
        x, g_y = generate_landmarks(cap=cap, device=device, pad=50)

        g_y = g_y.unsqueeze(0)
        x = x.unsqueeze(0)

        # forward
        # Calculate average encoding vector for video
        # f_lm_compact = f_lm.view(-1, f_lm.shape[-4], f_lm.shape[-3], f_lm.shape[-2], f_lm.shape[-1]) #BxK,2,3,224,224
        # train G

        x_hat = G(g_y, e_hat)

        plt.clf()
        out1 = x_hat.transpose(1, 3)[0] / 255
        # for img_no in range(1,x_hat.shape[0]):
        #    out1 = torch.cat((out1, x_hat.transpose(1,3)[img_no]), dim = 1)
        out1 = out1.to(cpu).numpy()
        # plt.imshow(out1)
        # plt.show()

        # plt.clf()
        out2 = x.transpose(1, 3)[0] / 255
        # for img_no in range(1,x.shape[0]):
        #    out2 = torch.cat((out2, x.transpose(1,3)[img_no]), dim = 1)
        out2 = out2.to(cpu).numpy()
        # plt.imshow(out2)
        # plt.show()

        # plt.clf()
        out3 = g_y.transpose(1, 3)[0] / 255
        # for img_no in range(1,g_y.shape[0]):
        #    out3 = torch.cat((out3, g_y.transpose(1,3)[img_no]), dim = 1)
        out3 = out3.to(cpu).numpy()
        # plt.imshow(out3)
        # plt.show()

        cv2.imshow('fake', cv2.cvtColor(out1, cv2.COLOR_BGR2RGB))
        cv2.imshow('me', cv2.cvtColor(out2, cv2.COLOR_BGR2RGB))
        cv2.imshow('ladnmark', cv2.cvtColor(out3, cv2.COLOR_BGR2RGB))

        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
