import matplotlib.pyplot as plt
import torch

def plot_frames(batch_of_pred, batch_of_target, start_frame, end_frame, batch_idx):
    pred = batch_of_pred[batch_idx].detach().to(torch.device('cpu'))
    target = batch_of_target[batch_idx].detach().to(torch.device('cpu'))
    fig, axs = plt.subplots(2, num_frames, figsize=(2*num_frames, 4))
    for frame in range(start_frame, end_frame):
        axs[0, frame].imshow(target[frame,:,:], cmap="Greys")
        axs[0, frame].axis('off')
        axs[1, frame].imshow(pred[frame,:,:], cmap="Greys")
        axs[1, frame].axis('off')
    plt.savefig(f'frames_in_batch_{batch_idx}.png', dpi=120)

def plot_curve(loss):
    loss = loss.detach().to(torch.device('cpu')).squeeze()
    fig, axs = plt.subplots(1,1)
    axs.plot(loss)
    plt.savefig(f"loss_curve.png",dpi=120)


def main():
    data = torch.rand((64,51,64,64))
    pred = torch.rand((64,51,64,64))
    error = torch.randn((100,1)) + torch.arange(100).unsqueeze(1)
    # plot_frames(pred, data, 10, 20, 6)
    # plot_curve(error)


if __name__ == "__main__":
    main()