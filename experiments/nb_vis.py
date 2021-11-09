import math
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms, utils
from IPython.display import display, update_display


to_pil = transforms.ToPILImage()


def display_batch(images, n_cols=4):
    images = images.cpu().detach()
    images = torch.minimum(images, torch.ones(images.shape))
    images = torch.maximum(images, torch.zeros(images.shape))

    fig = plt.figure(figsize=(n_cols * 3, len(images) * 3 // n_cols))
    for i, img in enumerate(images):
        ax = fig.add_subplot(math.ceil(len(images) / n_cols), n_cols, i+1)
        ax.axis('off')
        if img.shape[0] == 1:
            plt.imshow(to_pil(img), cmap='gray', interpolation='none', aspect='auto')
        else:
            plt.imshow(to_pil(img), interpolation='nearest', aspect='auto')
    plt.subplots_adjust(hspace=0, wspace=0)

    # Need to explicitly display and get an id if we want to dynamically update it
    display_id = random.randint(0, 100000)
    display(fig, display_id=display_id)

    return fig, display_id


def update_displayed_batch(images, fig, display_id):
    images = images.cpu().detach()
    images = torch.minimum(images, torch.ones(images.shape))
    images = torch.maximum(images, torch.zeros(images.shape))

    for i, img in enumerate(images):
        fig.axes[i].images[0].set_data(to_pil(img))

    update_display(fig, display_id=display_id)


def compare_batches(images, reconstructions, fig=None, display_id=None, n_cols=4):
    b, c, h, w = images.shape
    combined_batch = torch.cat((images, reconstructions), axis=1).reshape(2 * b, c, h, w)

    if fig is not None and display_id is not None:
        update_displayed_batch(combined_batch, fig, display_id)
    else:
        return display_batch(combined_batch, n_cols)
