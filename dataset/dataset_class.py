import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import face_alignment
from network.utils import timer
from dataset.video_extraction_conversion import (
    select_frames,
    select_images_frames,
    generate_landmarks,
    generate_cropped_landmarks,
)
import pickle


def save_video(path, frame_mark, video_id):
    """
    Generates the landmarks for the face in each provided frame and saves the frames and the landmarks as a pickled
    list of dictionaries with entries {'frame', 'landmarks'}.
    :param path: Path to the output folder where the file will be saved.
    :param video_id: Id of the video that was processed.
    :param frames: List of frames to save.
    :param face_alignment: Face Alignment model used to extract face landmarks.
    """
    path.mkdir(exist_ok=True)

    save_path = path / f'{video_id}.vid'
    pickle.dump(frame_mark, open(save_path, 'wb'))


class VidDataSet(Dataset):
    @timer
    def __init__(self, K, path_to_mp4, new_path, device):
        self.K = K
        self.path_to_mp4 = path_to_mp4
        self.device = device
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, flip_input=False, device=self.device.type
        )
        self.new_path = new_path

    @timer
    def __len__(self):
        return len(list(Path(self.path_to_mp4).glob('**/*.mp4')))

    @timer
    def __getitem__(self, idx):
        vid_idx = idx
        if idx < 0:
            idx = self.__len__() + idx

        path = list(Path(self.path_to_mp4).glob('**/*.mp4'))[idx]
        frame_mark = select_frames(path, self.K)
        frame_mark = generate_landmarks(frame_mark, fa=self.fa)
        frame_mark = torch.from_numpy(np.array(frame_mark)).type(
            dtype=torch.float
        )  # K,2,224,224,3
        frame_mark = frame_mark.transpose(2, 4).to(self.device)  # K,2,3,224,224
        """
        comented out because it can be generated in training
        I will fave frame_mark and load it directly in the training

        g_idx = torch.randint(low=0, high=self.K, size=(1, 1))
        x = frame_mark[g_idx, 0].squeeze()
        g_y = frame_mark[g_idx, 1].squeeze()

        return frame_mark, x, g_y, vid_idx
        """
        save_video(Path(self.new_path), frame_mark, vid_idx)


class MyNewDataset(Dataset):
    @timer
    def __init__(self, path_to_images, device):
        self.path_to_images = path_to_images
        self.device = device

    @timer
    def __len__(self):
        return len(list(Path(self.path_to_images).glob('*')))

    @timer
    def __getitem__(self, idx):
        vid_idx = idx
        if idx < 0:
            idx = self.__len__() + idx
        path = list(Path(self.path_to_images).glob('*'))[idx]
        frame_mark = pickle.load(open(path, 'rb'))
        K = frame_mark.shape[0]
        g_idx = torch.randint(low=0, high=K, size=(1, 1))
        x = frame_mark[g_idx, 0].squeeze()
        g_y = frame_mark[g_idx, 1].squeeze()

        return frame_mark, x, g_y, vid_idx


class FineTuningImagesDataset(Dataset):
    @timer
    def __init__(self, path_to_images, device):
        self.path_to_images = path_to_images
        self.device = device
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, flip_input=False, device=self.device.type
        )

    @timer
    def __len__(self):
        return len(list(Path(self.path_to_images).glob('*')))

    @timer
    def __getitem__(self, idx):
        frame_mark_images = select_images_frames(self.path_to_images)
        random_idx = torch.randint(low=0, high=len(frame_mark_images), size=(1, 1))
        frame_mark_images = [frame_mark_images[random_idx]]
        frame_mark_images = generate_cropped_landmarks(
            frame_mark_images, pad=50, fa=self.fa
        )
        frame_mark_images = torch.from_numpy(np.array(frame_mark_images)).type(
            dtype=torch.float
        )  # 1,2,256,256,3
        frame_mark_images = frame_mark_images.transpose(2, 4).to(
            self.device
        )  # 1,2,3,256,256

        x = frame_mark_images[0, 0].squeeze()
        g_y = frame_mark_images[0, 1].squeeze()

        return x, g_y


class FineTuningVideoDataset(Dataset):
    @timer
    def __init__(self, path_to_video, device):
        self.path_to_video = path_to_video
        self.device = device
        self.fa = fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, flip_input=False, device=self.device.type
        )

    @timer
    def __len__(self):
        return 1

    @timer
    def __getitem__(self, idx):
        path = self.path_to_video
        frame_has_face = False
        while not frame_has_face:
            try:
                frame_mark = select_frames(path, 1)
                frame_mark = generate_cropped_landmarks(frame_mark, pad=50, fa=self.fa)
                frame_has_face = True
            except:
                print('No face detected, retrying')
        frame_mark = torch.from_numpy(np.array(frame_mark)).type(
            dtype=torch.float
        )  # 1,2,256,256,3
        frame_mark = frame_mark.transpose(2, 4).to(self.device)  # 1,2,3,256,256

        x = frame_mark[0, 0].squeeze()
        g_y = frame_mark[0, 1].squeeze()
        return x, g_y
