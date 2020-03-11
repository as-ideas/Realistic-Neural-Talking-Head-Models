import torch


# device = torch.device("cpu")
device = torch.device("cpu")
cpu = torch.device("cpu")

path_to_chkpt = 'model_weights.tar'
path_to_backup = 'backup_model_weights.tar'

path_to_e_hat_video = 'e_hat_video.tar'
path_to_e_hat_images = 'e_hat_images.tar'

path_to_video = 'examples/fine_tuning/test_video.mp4'
path_to_images = 'examples/fine_tuning/test_images'
path_to_save = 'finetuned_model.tar'

VGGFace_body_path = 'Pytorch_VGGFACE_IR.py'
VGGFace_weight_path = 'Pytorch_VGGFACE.pth'

# Number of samples to be taken from each video
T = 32
num_epochs = 1
