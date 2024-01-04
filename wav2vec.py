import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import os
import fnmatch
import mne
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from braindecode.models import EEGConformer
import random
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from models.vq_vae import VQVAE

class EEGGPT(torch.nn.Sequential):
    # Join EEGConformer and GPT2LMHeadModel
    def __init__(self, w2v_model, vqvaq_model, eegconformer, gpt2lmheadmodel):
        super(EEGGPT, self).__init__(w2v_model, vqvaq_model, eegconformer, gpt2lmheadmodel)
        self.w2v_model = w2v_model
        self.vqvaq_model = vqvaq_model
        self.eegconformer = eegconformer
        self.gpt2lmheadmodel = gpt2lmheadmodel
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        # self.loss = self.vqvaq_model.loss
    
    def forward(self, x):
        x = self.w2v_model(x)
        x = self.encoder_layer(x.extract_features)
        x = x.unsqueeze(0)
        print(x.shape)
        result = self.vqvaq_model(x)
        # print(quantized_inputs)
        # print(input)
        x = x[:, :, :3]
        self.loss = self.vqvaq_model.loss_function(*result, M_N = 0.005)
        print(quantized_inputs.shape)
        
        x = torch.argmin(quantized_inputs, dim=2)
        # x = torch.argmax(x, dim=1)
        x = self.gpt2lmheadmodel(x, labels=x)
        x = x.logits.squeeze(0)
        # x = x.logits
        return x

def main():
    # W2V2 feature extractor
    w2v_model_name = "facebook/wav2vec2-base-960h"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(w2v_model_name)
    w2v_model = Wav2Vec2Model.from_pretrained(w2v_model_name)

    # GPT2
    gpt_model_name = 'gpt2'
    gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)


    # EEG-conformer
    eeg_model = EEGConformer(
        n_outputs=1024, 
        n_chans=12,
        filter_time_length=25, 
        pool_time_length=8,
        final_fc_length=1280,
    )

    vqvaq_model = VQVAE(in_channels=1, embedding_dim=512, num_embeddings=256)

    # Model
    model = EEGGPT(w2v_model, vqvaq_model, eeg_model, gpt_model)
    
    # CUDA
    model.cuda()

    input_audio, sample_rate = librosa.load("CantinaBand60_200.wav",  sr=200)

    i = feature_extractor(input_audio, return_tensors="pt", sampling_rate=16000)

    out = model(i.input_values.cuda())

    print(out.shape)

    text = tokenizer.decode(out[0], skip_special_tokens=False)

    print(text)

if __name__ == "__main__":
    main()
