from os import walk, getcwd
from os.path import join
import json

import matplotlib.pyplot as plt


def plot_losses(data, positions):
    plt.figure(1)

    for ind, (set_type, losses) in enumerate(data.items()):
        if set_type == "optim":
            title = "Learning rates testing"
        else:
            title = f"{set_type.capitalize()} dataset testing"
        plt.subplot(positions[ind])

        for model_type, valid_losses in losses.items():
            plt.plot(list(range(len(valid_losses))), valid_losses, label=model_type)
        
        plt.xlabel("Epoch")
        plt.ylabel('Loss')
        plt.title(title)
        plt.grid(True)
        leg = plt.legend(loc="upper left", ncol=1, mode=None, shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.5)
    
    plt.show()


def plot_final():
    path = "experiments/final/encdec_attn-final_losses.json"
    with open(path, 'r') as json_file:
        data = json.load(json_file)

    plt.figure(1)

    plt.subplot(121)
    for ind, losses in enumerate([data["train_losses"], data["valid_losses"]]):
        if ind == 0:
            lbl= "train"
        else:
            lbl = "valid"
        plt.plot(list(range(len(losses))), losses, label=lbl)
    plt.xlabel("Epoch")
    plt.ylabel('Loss')
    plt.title("Train vs valid losses")
    plt.grid(True)
    leg = plt.legend(loc="upper left", ncol=1, mode=None, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.subplot(122)
    for ind, losses in enumerate([data["train_ppls"], data["valid_ppls"]]):
        if ind == 0:
            lbl= "train"
        else:
            lbl = "valid"
        plt.plot(list(range(len(losses))), losses, label=lbl)
    plt.xlabel("Epoch")
    plt.ylabel('PPL')
    plt.title("Train vs valid PPLs")
    plt.grid(True)
    leg = plt.legend(loc="upper left", ncol=1, mode=None, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.show()


def get_losses(path):
    all_losses = {}

    for (dirpath, _, filenames) in walk(path):
        json_files = [file for file in filenames if "losses" in file and file.endswith(".json")]

        if len(json_files) > 0:
            key = dirpath.split('/')[-1].split('_')[0]
            key_losses = {}

            for file in json_files:
                with open(join(dirpath, file), 'r') as json_file:
                    data = json.load(json_file)
                    valid_losses = data["valid_losses"]
                    file_key = file.split('-')[-1].split('_')[0]

                    key_losses[file_key] = valid_losses
            
            all_losses[key] = key_losses
    
    return all_losses


if __name__ == "__main__":
    #plot_losses(get_losses(getcwd()), [131, 132, 133])

    plot_final()
