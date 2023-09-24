from torch.utils.data import Dataset
import pandas as pd

class MakeData(Dataset):

    def __init__(self, path, tokenizer) -> None:
        self.data = pd.read_csv(path, sep='\t', header=None)

        self.x = []

        for val in enumerate(self.data[0]):
            self.x.append('<SOS>: ' + val[1].split(',')[0] + ' <BOT>:' + val[1].split(',')[1] + '<EOS>')

        # print(self.x[0])

        self.x_encoded = tokenizer(self.x, max_length=30, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.x_encoded['input_ids']
        # print(self.input_ids[1])
        # print(type(self.input_ids[1]))
        self.attention_mask = self.x_encoded['attention_mask']
        # print(self.attention_mask[1])
        # print(type(self.attention_mask[1]))

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        # print(type(self.input_ids[index]))
        # print(type(self.attention_mask[index]))
        # print(type((self.input_ids[index], self.attention_mask[index])))
        return (self.input_ids[index], self.attention_mask[index])