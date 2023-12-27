import os
import fnmatch
import mne
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from braindecode.models import EEGConformer
import random
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
        return torch.tensor(data, dtype=torch.float32)

class EDFDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, collate_fn=None):
        super(EDFDataLoader, self).__init__(dataset, batch_size, shuffle, collate_fn)

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
        # save the preprocessed data as pickle file
        print(f"Saving {fname}.pt")
        torch.save(raw_data, f"{output_dir}/{fname}.pt")

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
    starting_point = random_starting_point(sample.shape[1])
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

    base_dir = "/media/kenneth/gujiga/eeg_tuf/edf/train/aaaaatvr"
    output_dir = "./preprocessed_data/"

    preprocess = False

    # Step 0: preprocess the data (optional)
    if preprocess == True:
        # create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        # preprocess the data
        preprocess_data(base_dir, output_dir)

    # Step 1: load the data
    dataset = EDFDataset(output_dir)

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

        for sample in dataset:
            print("loading data...")
            train_sample = sample_to_chunks(sample)
            # train_sample = train_sample.transpose(1, 2)
            # print(train_sample.shape)
            train_sample = torch.tensor(train_sample, dtype=torch.float32)
            train_sample = train_sample.cuda()

            # Step 2: create the model
            
            # Step 3: train the model
            model.train()

            # lengths = torch.tensor([500]*32)

            out = model(train_sample)
            # torch.nn.Softmax(dim=1)(out)
            # seq = torch.argmax(out, dim=1)

            # out_gpt = gpt_model(seq, labels=seq)
        
            # seq_gpt = torch.argmax(out_gpt, dim=1)
        
            optimizer.zero_grad()
            out.loss.backward()
            optimizer.step()

            print(out.loss)
        
        

        
        





if __name__ == "__main__":
    main()
