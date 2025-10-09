import matplotlib.pyplot as plt
import numpy as np
def imshow(img,title):
    """
    Display a single image tensor with a title.

    This function denormalizes the input tensor (assuming
    values were normalized around mean=0.5, std=0.5),
    converts it to a NumPy array, and displays it with matplotlib.

    Parameters
    ----------
    img : torch.Tensor
        Image tensor of shape (C, H, W) normalized to [-1, 1].
    title : str
        Title to display above the image.

    Returns
    -------
    None
        Displays the image using matplotlib.

    See Also
    --------
    matplotlib.pyplot.imshow : Display an image.
    torchvision.transforms.Normalize : Normalization used in preprocessing.

    Examples
    --------
    >>> images, labels = next(iter(train_loader))
    >>> imshow(images[0], "Example image")
    """

    img = img/2+0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.axis("off")
