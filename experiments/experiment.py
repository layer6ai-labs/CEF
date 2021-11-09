import math
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import utils
from tqdm import tqdm

from nb_vis import compare_batches, display_batch, update_displayed_batch


# Configure file system
notebook_path = Path(__file__).parent
project_root = notebook_path.parent
data_path = project_root / 'data'

os.chdir(project_root)

# Configure PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def train_injective_flow(flow, optim, scheduler, weight_schedule, train_loader, val_loader,
                         model_name, checkpoint_path, checkpoint_frequency=50, checkpoint_epoch=-1,
                         notebook_vis=True, num_recon_vis=4, num_gen_vis=8):
    '''Run the training loop for a given injective flow model'''
    embedding = flow.embedding
    base_flow = flow.distribution
    conformal = flow.conformal

    # Load from a previous checkpoint if specified
    if checkpoint_epoch >= 0:
        checkpoint = torch.load(checkpoint_path / f'{model_name}-e{checkpoint_epoch}.pt')

        flow.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        os.makedirs(checkpoint_path, exist_ok=True)

    # Initialize visualizations for notebook
    groundtruth_sample, _ = next(iter(val_loader))
    groundtruth_sample = groundtruth_sample[:num_recon_vis].to(device)

    def training_vis_ims():
        '''Helper to produce ground truth, reconstruction, and sample images for visualization'''
        with torch.no_grad():
            mid_latent_sample, _ = embedding.forward(groundtruth_sample)
            recon_sample, _ =  embedding.inverse(mid_latent_sample)

            gen_sample = flow.sample(num_gen_vis)

        return groundtruth_sample, recon_sample, gen_sample

    if notebook_vis:
        groundtruth_sample, recon_sample, gen_sample = training_vis_ims()
        fig1, display_id1 = compare_batches(groundtruth_sample, recon_sample)
        fig2, display_id2 = display_batch(gen_sample)

    # Training loop
    for epoch, (alpha, beta) in enumerate(weight_schedule(), checkpoint_epoch+1):

        flow.train()
        progress_bar = tqdm(enumerate(train_loader))

        for batch, (image, _) in progress_bar:
            image = image.to(device)
            optim.zero_grad()

            if alpha > 0 and beta > 0:
                assert conformal, 'Only CEFs admit joint training'

            # Compute reconstruction error
            with torch.set_grad_enabled(beta > 0):
                mid_latent, _ = embedding.forward(image)
                reconstruction, log_conf_det = embedding.inverse(mid_latent)
                reconstruction_error = torch.mean((image - reconstruction)**2)

            # Compute log likelihood
            with torch.set_grad_enabled(alpha > 0):
                log_pu = base_flow.log_prob(mid_latent)
                if conformal:
                    log_likelihood = torch.mean(log_pu - log_conf_det)
                else:
                    log_likelihood = torch.mean(log_pu)

            # Training step
            loss = - alpha*log_likelihood + beta*reconstruction_error
            loss.backward()
            optim.step()

            # Display results
            progress_bar.set_description(f'[E{epoch} B{batch}] | loss: {loss: 6.2f} | '
                                         f'LL: {log_likelihood:6.2f} '
                                         f'| recon: {reconstruction_error:6.5f} ')
            if notebook_vis and (batch + 1) % 10 == 0:
                groundtruth_sample, recon_sample, gen_sample = training_vis_ims()
                compare_batches(groundtruth_sample, recon_sample, fig1, display_id1)
                update_displayed_batch(gen_sample, fig2, display_id2)

        # Evaluate on the validation set
        if val_loader is not None:
            flow.eval()
            recon_errors = []
            log_likelihoods = []

            for image, _ in val_loader:
                image = image.to(device)

                with torch.no_grad():
                    reconstruction, log_likelihood = flow.reconstruct_and_log_prob(image)
                    reconstruction_error = torch.mean((image - reconstruction)**2, dim=(1, 2, 3))

                    recon_errors.extend(reconstruction_error.tolist())
                    log_likelihoods.extend(log_likelihood.tolist())

            val_recon_error = np.mean(recon_errors)
            val_log_likelihood = np.mean(log_likelihoods)
            val_loss = -alpha*val_log_likelihood + beta*val_recon_error
            print(f'[E{epoch} val] | loss: {val_loss} | '
                  'LL: {val_log_likelihood} | recon: {val_recon_error}')

        # Save a checkpoint
        if (epoch + 1) % checkpoint_frequency == 0:
            checkpoint = {
                'epoch': epoch,
                'alpha': alpha,
                'beta': beta,
                'reconstruction_error': val_recon_error,
                'log_likelihood': val_log_likelihood,
                'model': flow.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path / f'{model_name}-e{epoch}.pt')


def save_samples(flow, num_samples, gen_path, checkpoint_epoch=None, batch_size=8):
    '''Helper for generating samples from a trained model and saving to a folder'''

    if checkpoint_epoch is not None and checkpoint_epoch >= 0:
        checkpoint = torch.load(checkpoint_path / f'{model_name}-e{checkpoint_epoch}.pt')
        flow.load_state_dict(checkpoint['model'])

    os.makedirs(gen_path, exist_ok=True)

    for i in range(0, num_samples, batch_size):
        with torch.no_grad():
            gen_samples = flow.sample(min(batch_size, num_samples-i))

        for j, samp in enumerate(gen_samples):
            utils.save_image(gen_samples[j], f'{gen_path}/img{i+j}.jpg')
            print(f'Saved image {i+j}', end='\r')
