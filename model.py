import os
import fnmatch
import mne
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from braindecode.models import EEGConformer
import random
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from torchaudio.models import Conformer

def find_edf_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if fnmatch.fnmatch(file, '*.edf'):
                yield os.path.join(root, file)

def get_channels(data, nchannels):
    return data[:nchannels, :]

def resample(data, sample_rate=250):
    return mne.filter.resample(data.numpy(), sample_rate)

def read_edf_file(file_path):
    
    data = mne.io.read_raw_edf(file_path, verbose=False)
    fnmae = os.path.basename(file_path)
    raw_data = data.get_data()
    info = data.info

    channels = data.ch_names

    selected_data = get_channels(raw_data, 8)
    selected_data = torch.from_numpy(selected_data)

    return selected_data, info, channels, fnmae


def find_pt_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if fnmatch.fnmatch(file, '*.pt'):
                yield os.path.join(root, file)

def load_pt_file(file_path):
    return torch.load(file_path)

class EDFDataset(Dataset):
    def __init__(self, directory):
        self.files = list(find_pt_files(directory))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = load_pt_file(self.files[idx])
        data = torch.as_tensor(data, dtype=torch.float32)
        data = sample_to_chunks(data)
        data = torch.as_tensor(data, dtype=torch.float32)
        return data

class EDFDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, collate_fn=None, *args, **kwargs):
        super(EDFDataLoader, self).__init__(dataset, batch_size, shuffle, collate_fn, *args, **kwargs)

def preprocess_data(base_dir, output_dir):
    """
    Preprocesses the data by resampling it to 250 Hz and saving it as a pickle file
    """
    # find all edf files
    sample_files = list(find_edf_files(base_dir))
    print(f"Found {len(sample_files)} files")
    # preprocess each file
    for file in tqdm(sample_files):
        raw_data, info, channels, fname = read_edf_file(file)
        sample_rate = info['sfreq']
        # resample the data if the sample rate is not 250
        if sample_rate != 250:
            print(f"Resampling... {sample_rate}  ->  250")
            raw_data = resample(raw_data, sample_rate=250)
            print(raw_data.shape)

        # segment the data into 120 second chunks
        chunk_size = 120 * 250
        chunks = torch.split(torch.as_tensor(raw_data), chunk_size, dim=1)
    
        for i, data in enumerate(chunks[:-1]):
            # save the preprocessed data as pickle file
            print(f"Saving {output_dir}/{fname}_{i}.pt")
            torch.save(data, f"{output_dir}/{fname}_{i}.pt")

def batch_padding(batch):
    """
    Pads the batch with zeros to make the batch size uniform
    """
    max_len = max([len(x) for x in batch])
    padded_batch = []
    for sample in batch:
        padded_sample = torch.zeros(max_len)
        padded_sample[:len(sample)] = sample
        padded_batch.append(padded_sample)
    return padded_batch

def EDF_collate_fn(batch):
    return torch.stack(batch_padding(batch))



def sample_to_chunks(sample, chunk_length=2, overlap=0.2, padding_value=0):
    
    input_chunks = 32
    
    sampling_rate = 250
    chunk_length = 2 * sampling_rate
    overlap = int (0.2 * sampling_rate)

    # Randomly select starting points for each EEG recording
    starting_point = random_starting_point(sample.shape[1] - (input_chunks * chunk_length) - (input_chunks * overlap))
    starting_points = get_starting_points(starting_point, input_chunks, chunk_length, overlap)
    # Sample chunks from the EEG recording
    sampled_chunks = sample_chunks(sample, starting_points, input_chunks, chunk_length, overlap)

    return sampled_chunks


def random_starting_point(sample_length):
    """
    Randomly selects starting point
    """
    # randomly select one starting point
    starting_point = random.randint(0, sample_length)
    return starting_point
    
def get_starting_points(starting_point, input_chunks, chunk_length, overlap):
    """
    Returns a list of starting points for each chunk
    """
    # sample chunks
    starting_points = []
    for i in range(input_chunks):
        starting_points.append(starting_point + (i * chunk_length) - (i * overlap))
    
    return starting_points 
    
def sample_chunks(sample, starting_points, input_chunks, chunk_length, overlap):
    if sample.shape[1] < starting_points[0] + (input_chunks * chunk_length) - (input_chunks * overlap):
        # padding
        padded_sample = torch.zeros(sample.shape[0], (input_chunks * chunk_length) - (input_chunks * overlap))
        padded_sample = torch.vstack((sample, padded_sample))
    else:
        padded_sample = sample

    chunks = [padded_sample[:, starting_points[i]:starting_points[i] + chunk_length] for i in range(input_chunks)]
    #chunks list to tensor
    chunks = torch.stack(chunks)

    return chunks


class EEGGPT(torch.nn.Sequential):
    # Join EEGConformer and GPT2LMHeadModel
    def __init__(self, eegconformer, gpt2lmheadmodel):
        super(EEGGPT, self).__init__(eegconformer, gpt2lmheadmodel)
        self.eegconformer = eegconformer
        self.gpt2lmheadmodel = gpt2lmheadmodel
        # self.loss = self.gpt2lmheadmodel.loss
    
    def forward(self, x):
        x = self.eegconformer(x)
        x = torch.argmax(x, dim=1)
        x = self.gpt2lmheadmodel(x, labels=x)
        return x

def main():

    # Directories
    base_dir = "/media/kenneth/gujiga/eeg_tuf/edf/train/aaaaatvr"
    output_dir = "./train_segmented/"
    test_dir = "./test_data/"

    # Create model output directory if it doesn't exist
    model_dir = "./models/"
    os.makedirs(model_dir, exist_ok=True)

    # Config
    preprocess = False

    # Step 0: preprocess the data (optional)
    if preprocess == True:
        # create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        # preprocess the data
        preprocess_data(base_dir, output_dir)

    # Step 1: load the data
    dataset = EDFDataset(output_dir)
    # Dataloader
    dataloader = EDFDataLoader(dataset, batch_size=8, shuffle=True, num_workers=10)

    test_dataset = EDFDataset(test_dir)
    test_dataloader = EDFDataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=10)

    
    eeg_model = EEGConformer(
            n_outputs=1024, 
            n_chans=8,
            filter_time_length=25, 
            pool_time_length=8,
            final_fc_length=1280,
        )

    gpt_model_name = 'gpt2'
    gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model_name)
    
    # CUDA
    eeg_model.cuda()
    gpt_model.cuda()

    # Model
    model = EEGGPT(eeg_model, gpt_model)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for n_epochs in range(10):
        print(f"Epoch {n_epochs}")

        model.train()
        for sample in tqdm(dataloader):
            for s in sample:
                train_sample = s.cuda()
                # Step 2: zero the gradients
                optimizer.zero_grad()
                # Step 3: run the model
                out = model(train_sample)
                # Step 4: backpropagate the loss
                out.loss.backward()
                optimizer.step()
                # Step 5: print the loss
                print(f"Train loss: {out.loss}")

        if n_epochs % 5 == 0:
            # Step 6: save the model
            torch.save(model.state_dict(), f"models/model_{n_epochs}.pt")

        # Step 7: test the model
        # model.eval()
        # with torch.inference_mode():
        #     for sample in test_dataloader:
        #         test_sample = sample[0].cuda()
        #         out = model(test_sample)
        #         print(f"Test loss: {out.loss}")

if __name__ == "__main__":
    main()
