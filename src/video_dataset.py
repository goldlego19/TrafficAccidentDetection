"""
CADP Dataset Loader for 3D CNN
Path structure: data/cadp/extracted_frames/[videoid]/[frame_number].jpg
"""
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path

class TemporalAccidentDataset(Dataset):
    def __init__(self, df, frames_dir, sequence_length=16, transform=None, augment=False):
        self.df = df
        self.frames_dir = Path(frames_dir)
        self.sequence_length = sequence_length
        self.transform = transform
        self.augment = augment
        
        self.clips = []
        self._build_clip_index()
    def __len__(self):
        """Returns the total number of normal and abnormal clips identified"""
        return len(self.clips)

    def _build_clip_index(self):
        """Matches CSV rows to the CADP folder structure"""
        for _, row in self.df.iterrows():
            # Match your CSV column 'video' to folder names like '000001'
            vid_id = str(row['videoid']).zfill(6) 
            start_f = int(row['startframe'])
            end_f = int(row['endframe'])
            
            vid_folder = self.frames_dir / vid_id
            if not vid_folder.exists():
                continue

            # 1. Normal Clip: Driving before the 'startframe'
            if start_f >= self.sequence_length:
                self.clips.append({'vid_id': vid_id, 'start': 0, 'end': start_f, 'label': 0})
                
            # 2. Abnormal Clip: The accident segment
            if (end_f - start_f) >= 5: # Minimum frames for an accident clip
                self.clips.append({'vid_id': vid_id, 'start': start_f, 'end': end_f, 'label': 1})

    def __getitem__(self, idx):
        c = self.clips[idx]
        vid_folder = self.frames_dir / c['vid_id']
        
        # Load all available JPGs in that folder
        all_frames = sorted(list(vid_folder.glob('*.jpg')))
        
        # Slice frames based on the bounds identified in the index
        segment = all_frames[c['start']:c['end']]
        
        # Uniformly sample to reach exactly sequence_length
        sampled_paths = self._sample_frames(segment)
        
        frames = []
        for p in sampled_paths:
            img = cv2.imread(str(p))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
            
        # Convert to Tensor (Time, Channels, Height, Width)
        video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
        
        if self.transform:
            video = self.transform(video)
            
        # Final permute for R(2+1)D: (C, T, H, W)
        return video.permute(1, 0, 2, 3), torch.tensor([c['label']], dtype=torch.float32)

    def _sample_frames(self, segment):
        if len(segment) >= self.sequence_length:
            indices = np.linspace(0, len(segment) - 1, self.sequence_length).astype(int)
            return [segment[i] for i in indices]
        return list(segment) + [segment[-1]] * (self.sequence_length - len(segment))