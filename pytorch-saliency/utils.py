import torchvision 
import matplotlib.pyplot as plt 
from ipywidgets import interact, IntSlider
import numpy as np

def saliency_plot(x, cam=None, alpha=0.5, scale=1.0, nrow=None, sl=None, show_cb=True, save_fig=False, mode="3D", interactive=False):
    """ Plotting of batches and saliency maps.

    Parameters:
        x (torch.Tensor): Batch of single-color images to be plotted in greyscale.
            For mode="2D" must be of form [batch_size, channels, height, width].
            For mode="3D" must be of form [batch_size, channels, depth, height, width]
        cam (torch.Tensor): Argument to provide additional tensor for plotting.
            The tensor will be plotted over x with transparency level
            set by alpha.
        alpha (double): Control transparency level of cam overlay. Default: 0.5
        scale (double): Scale factor for plot size. Default: 1.0
        nrow (int): Control number of images displayed per row. Default: Batch size.
        sl (int): Select default slice to display (only relevant for mode="3D"). Default: Mid slice.
        show_cb (bool): Show colorbar in plot. Default: True
        save_fig (bool): If False plot will be outputted using plt.plot(). If
            True plot will be saved as "saliency.png"
        mode (str): Choose between "2D" and "3D" for two dimensional or three dimensional data respectively.
            Default:"3D"
        interactive (bool): Show ipywidget slider to cycle through slices (only relevant for mode="3D"). Default: False.
    """
    assert mode in ("2D", "3D"), f"{mode} is not a valid mode. Must be '2D' or '3D'."

    if nrow is None:
        nrow = x.shape[0]
    if sl is None:
        sl = x.shape[2]//2

    if mode == "2D":
        assert len(x.shape) == 4, "Input for 2D images must be of form [batch_size, channels, height, width]. For 3D data use mode='3D'."
        plt.figure(figsize=(6*scale, 4*scale))
        # make grid of size 1 x batchsize for plotting
        x = torchvision.utils.make_grid(x, nrow=nrow)
        npimg = x.detach().numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)))
        if cam is not None:
            cam = torchvision.utils.make_grid(cam, nrow=nrow)[0]
            plt.imshow(cam.detach().numpy(), cmap='viridis', alpha=alpha)
            if show_cb:
                im_ratio = cam.shape[0]/cam.shape[1]
                plt.colorbar(fraction=0.046*im_ratio, pad=0.04)
        if save_fig:
            plt.savefig("saliency.png")
        else:
            plt.show()
    else:
        assert len(x.shape) == 5, "Input for 3D images must be of form [batch_size, channels, depth, height, width]. For 2D data use mode='2D'."
        cam = [] if cam is None else cam
        cam = [cam] if not isinstance(cam, list) else cam

        def show(sl):
            # pylint: disable=unused-variable
            fig = plt.figure(figsize=(6*scale, 4*scale))
            plt.subplot(1+len(cam), 1, 1)
            # make grid of size 1 x batchsize for plotting
            print(x.shape)
            x_grid = torchvision.utils.make_grid(x, nrow=nrow)
            print(x_grid.shape)
            npimg = x_grid.detach().numpy()
            cmap = None
            print(npimg.shape)
            plt.imshow(np.transpose(npimg[:, sl, :, :], (1,2,0)), cmap=cmap)
            for j, cam_j in enumerate(cam):
                # pylint: disable=unused-variable
                ax = plt.subplot(1+len(cam), 1, j+2)
                plt.imshow(npimg[sl], cmap=cmap)
                cam_grid = torchvision.utils.make_grid(cam_j.squeeze(1), nrow=nrow)
                plt.imshow(cam_grid.detach().numpy()[sl], cmap='viridis', alpha=alpha)
                if show_cb:
                    im_ratio = cam_grid.shape[0]/cam_grid.shape[1]
                    plt.colorbar(fraction=0.046*im_ratio, pad=0.04)
            if save_fig:
                plt.savefig("saliency.png")
            else:
                plt.tight_layout()
                plt.show()

        if not interactive:
            show(sl)
        else:
            slider_var = IntSlider(min=0, max=x.shape[2]-1, step=1, value=sl, description="Slice:")
            interact(show, sl=slider_var)