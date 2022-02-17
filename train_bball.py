"""Main entry point of the code"""
from __future__ import print_function

import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import autograd
# torch.manual_seed(1997)

from networks import BallModel
# from model_components import GruState 
from argument_parser import argument_parser
from dataset import get_dataloaders
from logbook.logbook import LogBook
from utils.util import set_seed, make_dir
from utils.visualize import ScalarLog, VectorLog, HeatmapLog
from box import Box

import os
from os import listdir
from os.path import isfile, join

set_seed(1997)

loss_fn = torch.nn.BCELoss()

def repackage_hidden(ten_):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(ten_, torch.Tensor):
        return ten_.detach()
    else:
        return tuple(repackage_hidden(v) for v in ten_)

def nan_hook(_tensor):
        nan_mask = torch.isnan(_tensor)
        if nan_mask.any():
            raise RuntimeError(f"Found NAN in: ", nan_mask.nonzero(), "where:", _tensor[nan_mask.nonzero()[:, 0].unique(sorted=True)])

def get_grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def train(model, train_loader, optimizer, epoch, logbook,
          train_batch_idx, args):
    """Function to train the model"""
    grad_norm_log = ScalarLog(args.folder_log+'/intermediate_vars', "grad_norm", epoch=epoch)

    model.train()


    epoch_loss = torch.tensor(0.).to(args.device)
    for batch_idx, data in enumerate(train_loader):
        hidden = model.init_hidden(data.shape[0]).to(args.device) # NOTE initialize per epoch or per batch [??]

        start_time = time()
        data = data.to(args.device)
        if data.dim()==4:
            data = data.unsqueeze(2).float()
        hidden = hidden.detach()
        optimizer.zero_grad()
        loss = 0.
        
        for frame in range(49):
            output, hidden, intm = model(data[:, frame, :, :, :], hidden) # would it work? *_ ?

            target = data[:, frame + 1, :, :, :]
            loss += loss_fn(output, target)

        (loss).backward()
        grad_norm = get_grad_norm(model)
        grad_norm_log.append(grad_norm)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        grad_norm_log.save()

        train_batch_idx += 1 # TOTAL batch index
        metrics = {
            "loss": loss.cpu().item(),
            "mode": "train",
            "batch_idx": train_batch_idx,
            "epoch": epoch,
            "time_taken": time() - start_time,
        }
        logbook.write_metric_logs(metrics=metrics)

        epoch_loss += loss.detach()
        
    if args.log_intm_frequency > 0 and epoch % args.log_intm_frequency == 0:
        """log intermediate variables here"""
        pass
        # SAVE logged vectors

    epoch_loss = epoch_loss / (batch_idx+1)
    return train_batch_idx, epoch_loss.detach()

@torch.no_grad() 
def test(model, test_loader, epoch, transfer_loader, logbook, # TODO not adapted yet
         train_batch_idx, args):
    model.eval()
    batch = 0
    losses = []
    start_time = time()

    for data in test_loader:
        data = data.to(args.device)
        loss = 0
        ### Rollout a single trajectory for all frames, using the previous
        if args.should_save_csv and batch == 0:
            for trajectory_to_save in range(4):
                hidden = model.init_hidden(args.batch_size).to(args.device)
                for frame in range(25): # given reference
                    output, hidden = model(data[:, frame, :, :, :], hidden)
                    target = data[:, frame + 1, :, :, :]

                    np.savetxt(f"{args.folder_log}ROP_{epoch}_"
                               f"{trajectory_to_save}_{frame}.csv",
                               output[trajectory_to_save].cpu()
                               .detach().numpy().flatten(), delimiter=',')
                    np.savetxt(f"{args.folder_log}ROT_{epoch}_"
                               f"{trajectory_to_save}_{frame}.csv",
                               target[trajectory_to_save].cpu()
                               .numpy().flatten(), delimiter=',')
                for frame in range(25, 49): # completely recursive prediction
                    output, hidden  = model(output, hidden)
                    np.savetxt(f"{args.folder_log}ROP_{epoch}_"
                               f"{trajectory_to_save}_{frame}.csv",
                               output[trajectory_to_save].cpu()
                               .detach().numpy().flatten(), delimiter=',')
                    np.savetxt(f"{args.folder_log}ROT_{epoch}_"
                               f"{trajectory_to_save}_{frame}.csv",
                               target[trajectory_to_save].cpu()
                               .numpy().flatten(), delimiter=',')

        ### Save all frames from the first 9 trajectories
        hidden = model.init_hidden(args.batch_size).to(args.device)

        for frame in range(49):
            output, hidden = model(data[:, frame, :, :, :], hidden)
            target = data[:, frame + 1, :, :, :]
            loss = loss_fn(output, target)
            losses.append(loss.cpu().detach().numpy())
        batch += 1
        print("Test loss is: ", loss)

    logbook.write_metric_logs(metrics={
        "loss": np.sum(np.array(losses)).item(),
        "mode": "test",
        "epoch": epoch,
        "batch_idx": train_batch_idx,
        "time_taken": time() - start_time,
    })

    batch = 0
    losses = []
    start_time = time()

    for data in transfer_loader:
        data = data.to(args.device)
        loss = 0
        ### Rollout a single trajectory for all frames, using the previous
        if args.should_save_csv and batch == 0:
            for trajectory_to_save in range(9):
                hidden = model.init_hidden(args.batch_size).to(args.device)
                for frame in range(25):
                    output, hidden = model(data[:, frame, :, :, :], hidden)
                    target = data[:, frame + 1, :, :, :]
                    np.savetxt(f"{args.folder_log}ROPT_{epoch}_"
                               f"{trajectory_to_save}_{frame}.csv",
                               output[trajectory_to_save].cpu()
                               .detach().numpy().flatten(), delimiter=',')
                    np.savetxt(f"{args.folder_log}ROTT_{epoch}_"
                               f"{trajectory_to_save}_{frame}.csv",
                               target[trajectory_to_save].cpu()
                               .numpy().flatten(), delimiter=',')

                for frame in range(25, 49):
                    output, hidden = model(output, hidden)
                    target = data[:, frame + 1, :, :, :]
                    np.savetxt(f"{args.folder_log}ROPT_{epoch}_"
                               f"{trajectory_to_save}_{frame}.csv",
                               output[trajectory_to_save].cpu()
                               .detach().numpy().flatten(), delimiter=',')
                    np.savetxt(f"{args.folder_log}ROTT_{epoch}_"
                               f"{trajectory_to_save}_{frame}.csv",
                               target[trajectory_to_save].cpu()
                               .numpy().flatten(), delimiter=',')

        hidden = model.init_hidden(args.batch_size).to(args.device)

        for frame in range(49):
            output, hidden = model(data[:, frame, :, :, :], hidden)
            target = data[:, frame + 1, :, :, :]
            loss = loss_fn(output, target)
            losses.append(loss.cpu().detach().numpy())
        batch += 1
        print("Transfer loss is: ", loss)

    logbook.write_metric_logs(metrics={
        "loss": np.sum(np.array(losses)).item(),
        "mode": "transfer",
        "epoch": epoch,
        "batch_idx": train_batch_idx,
        "time_taken": time() - start_time,
    })
    if args.should_save_csv:
        np.savetxt(args.folder_log + 'losses_' +
                   str(epoch) + '.csv', np.array(losses), delimiter=',')

def main():
    """Function to run the experiment"""
    args = argument_parser()
    
    print(args)
    logbook = LogBook(config=args)

    if not args.should_resume:
        # New Experiment
        make_dir(f"{args.folder_log}/model")
        make_dir(f"{args.folder_log}/checkpoints")
        logbook.write_message_logs(message=f"Saving args to {args.folder_log}/model/args")
        torch.save({"args": vars(args)}, f"{args.folder_log}/model/args")

    cudable = torch.cuda.is_available()
    args.device = torch.device("cuda" if cudable else "cpu")
    
    model, optimizer, start_epoch, train_batch_idx, epoch_loss_log = setup_model(args=args, logbook=logbook)

    args.directory = './data' # dataset directory
    train_loader, test_loader, transfer_loader = get_dataloaders(args)

    for epoch in range(start_epoch, args.epochs + 1):
        train_batch_idx, epoch_loss = train(model=model,
                                train_loader=train_loader,
                                optimizer=optimizer,
                                epoch=epoch,
                                logbook=logbook,
                                train_batch_idx=train_batch_idx,
                                args=args)
        epoch_loss_log.append(epoch_loss)
        epoch_loss_log.save()

        # TODO test
        # if epoch%50==0:
        #     print("Epoch number", epoch)
        #     test(model=model,
        #         test_loader=test_loader,
        #         epoch=epoch,
        #         transfer_loader=transfer_loader,
        #         logbook=logbook,
        #         train_batch_idx=train_batch_idx,
        #         args=args)

        if args.model_persist_frequency > 0 and epoch % args.model_persist_frequency == 0:
            logbook.write_message_logs(message=f"Saving model to {args.folder_log}/checkpoints/{epoch}")
            torch.save(model.state_dict(), f"{args.folder_log}/model/{epoch}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'epoch_loss_log': epoch_loss_log,
                'train_batch_idx': train_batch_idx
            }, f"{args.folder_log}/checkpoints/{epoch}")


def setup_model(args, logbook):
    """Function to setup the model"""

    model = BallModel(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 1
    train_batch_idx = 0
    epoch_loss_log = ScalarLog(args.folder_log+'/intermediate_vars', "epoch_loss")
    if args.should_resume:
        # Find the last checkpointed model and resume from that
        model_dir = f"{args.folder_log}/checkpoints"
        latest_model_idx = max(
            [int(model_idx) for model_idx in listdir(model_dir)
             if model_idx != "args"]
        )
        args.path_to_load_model = f"{model_dir}/{latest_model_idx}"
        args.checkpoint = {"epoch": latest_model_idx}

    if args.path_to_load_model != "":
        checkpoint = torch.load(args.path_to_load_model.strip(), map_location=args.device) 
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        epoch_loss_log = checkpoint['epoch_loss_log']
        train_batch_idx = checkpoint['train_batch_idx']

        logbook.write_message_logs(message=f"Resuming experiment id: {args.id}, from epoch: {start_epoch}")

    return model, optimizer, start_epoch, train_batch_idx, epoch_loss_log


if __name__ == '__main__':
    main()
