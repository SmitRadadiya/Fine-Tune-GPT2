import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils import MakeData
from tqdm import tqdm
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def training(data, model, optimizer):

    num_epochs = 2
    loss_val = []
    for epoch in range(num_epochs):
        loop = tqdm(data)
        for i, (x, a) in enumerate(loop):
            x = x.to(device)
            a = a.to(device)
            optimizer.zero_grad()
            loss = model(x, attention_mask = a, labels = x).loss
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss = loss.item())
        loss_val.append(loss.item())
        torch.save(model.state_dict(), f'checkpoint/model_state{num_epochs}.pt')
    plt.plot(loss_val)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('loss_plot.png')
    plt.show()



tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '<PAD>',
                              'bos_token': '<SOS>',
                              'eos_token': '<EOD>'})
tokenizer.add_tokens(['<BOT>:'])

model = GPT2LMHeadModel.from_pretrained('gpt2') 
model.resize_token_embeddings(len(tokenizer))

model = model.to(device)

# inp = 'Hi How are you!'
# op = tokenizer.decode(model.generate(**tokenizer(inp, return_tensors='pt'))[0])
# print(op)


def main():
    data = MakeData('data/Assignment2aDataset.txt', tokenizer)
    train_loader = DataLoader(data, batch_size=64)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    training(train_loader, model, optimizer)
    infer()

def infer():
    model_test = torch.load('checkpoint/model_state30.pt')
    model.load_state_dict(model_test)
    model.eval()
    print("Press q for quit...")
    while True:
        inp = input('Date: ')
        if inp == 'q':
            break
        inp = '<SOS>: ' + inp + ' <BOT>:' 
        inp = tokenizer(inp, return_tensors="pt")
        x = inp['input_ids'].to(device)
        a = inp['attention_mask'].to(device)
        op = model.generate(x, attention_mask=a, max_new_tokens=10)
        op = tokenizer.decode(op[0])
        print(op)

if __name__ == '__main__':
    main()

