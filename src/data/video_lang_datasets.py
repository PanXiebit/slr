from torch.utils.data import Dataset, DataLoader
from src.data.vocabulary import Vocabulary
import logging, os
import pandas as pd
import torch
import glob, struct
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
import random

class PhoenixVideo(Dataset):
    def __init__(self, vocab_file, corpus_dir, video_path, phase, DEBUG=False):
        """
        :param phase:  'train', 'dev', 'test'
        """
        self.vocab_file = vocab_file
        self.image_type = 'png'
        self.max_video_len = 300
        self.corpus_dir = corpus_dir
        self.video_path = video_path
        self.phase = phase
        self.sample = True
        self.input_shape = 112

        self.alignment = {}
        self.vocab = Vocabulary(self.vocab_file)
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomCrop(self.input_shape),
            transforms.ToTensor(),
            normalize,]
        )
        self.test_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.CenterCrop(self.input_shape),
            transforms.ToTensor(),
            normalize,
        ])

        self.phoenix_dataset = self.load_video_list()
        self.data_dict = self.phoenix_dataset[phase]
        if DEBUG == True:
            self.data_dict = self.data_dict[:101]

        logging.info('[DATASET: {:s}]: total {:d} samples.'.format(phase, len(self.data_dict)))

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        cur_vid_info = self.data_dict[idx]
        id = cur_vid_info['id']
        frames_list = self.get_images(cur_vid_info['path'])
        label = cur_vid_info['label']
        data_len = len(frames_list)   # frame number
        sample = {'id': id, 'data': frames_list, 'label': label, "data_len": data_len}
        return sample

    def load_video_list(self):
        phoenix_dataset = {}
        outliers = ['13April_2011_Wednesday_tagesschau_default-14'] # '05July_2010_Monday_heute_default-8'
        for task in ['train', 'dev', 'test']:
            if task != self.phase:
                continue
            dataset_path = os.path.join(self.video_path, task)
            corpus = pd.read_csv(os.path.join(self.corpus_dir, '{:s}.corpus.csv'.format(task)), sep='|')
            videonames = corpus['folder'].values
            annotation = corpus['annotation'].values
            ids = corpus['id'].values
            num_sample = len(ids)
            video_infos = []
            for i in range(num_sample):
                if ids[i] in outliers:
                    continue
                tmp_info = {
                    'id': ids[i],
                    'path': os.path.join(self.video_path, task, videonames[i].replace('*.png', '')),
                    'label_text': annotation[i],
                    'label': self.sentence2index(annotation[i])
                }
                video_infos.append(tmp_info)
            phoenix_dataset[task] = video_infos
        return phoenix_dataset

    def sentence2index(self, sent):
        sent = sent.split(' ')
        s = []
        for word in sent:
            if word in self.vocab.word2index:
                s.append(self.vocab.word2index[word])
            else:
                s.append(self.vocab.word2index['<UNK>'])
        return s

    def load_video(self, video_name):
        feat = caffeFeatureLoader.loadVideoC3DFeature(video_name, 'pool5')
        feat = torch.tensor(feat)
        return feat

    def get_images(self, video_name):
        frames_list = glob.glob(os.path.join(video_name, '*.{:s}'.format(self.image_type)))
        frames_list.sort()
        num_frame = len(frames_list)
        if self.phase == 'train' and self.sample and num_frame > self.max_video_len:
            # first, Randomly repeat 20%. Second, Randomly delete 20%
            ids = list(range(num_frame))
            add_idx = random.sample(ids, int(0.2 * len(ids)))
            ids.extend(add_idx)
            ids.sort()
            ids = random.sample(ids, int(0.8 * len(ids)))
            ids.sort()
            if len(ids) > self.max_video_len:
                ids = random.sample(ids, self.max_video_len)
                ids.sort()
            frames_list = [frames_list[i] for i in ids]
        return frames_list

    def load_video_from_images(self, frames_list):
        frames_tensor_list = [self.load_image(frame_file, self.phase) for frame_file in frames_list]
        video_tensor = torch.stack(frames_tensor_list, dim=0)
        return video_tensor

    def load_image(self, img_name, phase, reduce_mean=True):
        image = Image.open(img_name)
        if phase == "train" :
            image = self.transform(image)
        elif phase == "test" or phase == "dev":
            image = self.test_transform(image)
        return image

    def collate_fn_video(self, batch, padding=6):
        # batch.sort(key=lambda x: x['data'].shape[0], reverse=True)
        len_video = [x["data_len"] for x in batch]
        len_label = [len(x['label']) for x in batch]
        batch_video = torch.zeros(len(len_video), max(len_video), 3, self.input_shape, self.input_shape)  # padding with zeros
        batch_decoder_label = torch.zeros(len(len_video), max(len_label) + 2).long()  # [batch, max_len_label]
        batch_label = []
        IDs = []
        len_decoder_label = []
        for i, bat in enumerate(batch):
            data = self.load_video_from_images(bat['data'])
            label = bat['label']
            len_decoder_label.append(len_label[i] + 2)
            batch_label.extend(label)
            batch_decoder_label[i, 1:1+len(label)] = torch.LongTensor(label)
            batch_decoder_label[i, 0] = self.vocab.bos()   # bos
            batch_decoder_label[i, 1+len(label)] = self.vocab.eos() # eos
            batch_video[i, :len_video[i], :] = torch.FloatTensor(data)
            IDs.append(bat['id'])
        batch_label = torch.LongTensor(batch_label)
        batch_decoder_label = torch.LongTensor(batch_decoder_label)
        len_video = torch.LongTensor(len_video)
        len_label = torch.LongTensor(len_label)
        len_decoder_label = torch.LongTensor(len_decoder_label)

        # batch_video = batch_video.permute(0, 2, 1)

        return {'data': batch_video, 'label': batch_label, 'decoder_label': batch_decoder_label,
                'len_data': len_video, 'len_label': len_label, 'len_decoder_label':len_decoder_label,
                'id': IDs}


class caffeFeatureLoader():
    @staticmethod
    def loadVideoC3DFeature(sample_name, feattype = 'pool5'):
        featnames = glob.glob(os.path.join(sample_name, '*.' + feattype))
        featnames.sort()
        feat = []
        for name in featnames:
            feat.append(caffeFeatureLoader.loadC3DFeature(name)[0])
        return feat

    @staticmethod
    def loadC3DFeature(filename):
        feat = []
        with open(filename, 'rb') as fileData:
            num = struct.unpack("i", fileData.read(4))[0]
            chanel = struct.unpack("i", fileData.read(4))[0]
            length = struct.unpack("i", fileData.read(4))[0]
            height = struct.unpack("i", fileData.read(4))[0]
            width = struct.unpack("i", fileData.read(4))[0]
            blob_shape = [num, chanel, length, height, width]
            m = num * chanel * length * height * width
            for i in range(m):
                val = struct.unpack("f", fileData.read(4))[0]
                feat.append(val)
        return feat, blob_shape

if __name__ == "__main__":
    vocab_file = "/workspace/full_conv/Data/slr-phoenix14/newtrainingClasses.txt"
    corpus_dir = "/workspace/full_conv/Data/slr-phoenix14"
    video_path = "/workspace/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px"
    phase = "train"
    train_datasets = PhoenixVideo(vocab_file, corpus_dir, video_path, phase, DEBUG=True)
    print(train_datasets[0].keys(), train_datasets[0]["data"], train_datasets[0]["data_len"])

    train_iter = DataLoader(train_datasets,
               batch_size=1,
               shuffle=True,
               num_workers=2,
               collate_fn=train_datasets.collate_fn_video,
               drop_last=True)

    for i, batch in enumerate(train_iter):
        if i >0:
            break
        print(batch.keys())
        print(batch["data"].shape)
        print(batch['label'])
        print(batch['decoder_label'])
        print(batch['len_label'])
        print(batch['len_decoder_label'])
