import torch.nn as nn
from torch.nn import functional
import torch.optim as optim
from model import PoemLSTM
from data_loader import *
from torch.utils.data import DataLoader
from tqdm import tqdm



def train_model(model, dataloader, optimizer_type, loss_function, learning_rate):

    model.train()

    optimizer = optimizer_type(params=model.parameters(), lr=learning_rate)
    mean_l = 0

    for batch in dataloader:
        topic = batch[0]
        input_sent = batch[1]
        targ_sent = batch[2]

        # forward pass
        output = model(topic=topic, inp=input_sent)

        # calculate the loss and perform backprop
        loss = loss_function(output, targ_sent)
        mean_l += loss.item()

        # zero grad
        optimizer.zero_grad()

        loss.backward()

        # update parameters
        optimizer.step()

    return mean_l/len(dataloader)


def evaluate_model(model,dataloader,loss_function):

    model.eval()

    mean_l = 0
    with torch.no_grad():
        for batch in dataloader:
            topic = batch[0]
            input_sent = batch[1]
            targ_sent = batch[2]

            # forward pass
            output = model(topic=topic, inp=input_sent)

            # loss
            loss = loss_function(output, targ_sent)
            mean_l += loss.item()

    return mean_l/len(dataloader)


if __name__ == "__main__":

    # Initialize Models
    vocabulary,topic_voc = train_dataset.build_vocab()
    hidden_size = 256
    vocab_size = len(vocabulary)
    sent_embedding_dim = 300

    topic_embedding_dim = 100
    topic_vocab_size= len(topic_voc)
    topic_hidden_size = 100
    # print(top_vocab_size)
    PoemGenerator = PoemLSTM(hidden_size, vocab_size, sent_embedding_dim, topic_embedding_dim, topic_vocab_size, topic_hidden_size)

    # Set Hyperparameter Values
    NUM_EPOCHS = 17
    LEARNING_RATE = 0.001
    OPTIMIZER = optim.Adam
    LOSS_FUNCTION = nn.functional.cross_entropy

    train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

    # train
    tr_losses = []
    for epoch in tqdm(range(NUM_EPOCHS)):
        tr_loss = train_model(model=PoemGenerator, dataloader=train_dataloader, optimizer_type=OPTIMIZER, loss_function=LOSS_FUNCTION, learning_rate=LEARNING_RATE)
        print("Training loss per epoch",tr_loss)
        tr_losses.append(tr_loss)
        test_loss = evaluate_model(model=PoemGenerator, dataloader=test_dataloader, loss_function=LOSS_FUNCTION)
        print("Test loss per epoch", test_loss)


