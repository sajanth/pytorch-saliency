# Caveats
* Current implementation is only suitable for networks whoes architecture is cleary divided into two named sections corresponding to a 
convolutional and feed forward part
* vgg16 ✔️
* resnet18 ❌ 
# Saliency maps in NIP

In the following we discuss the Guided Backpropagation[^2] and Grad-CAM[^6] method for the generation of saliency maps. Here we use the term "saliency map" in a very broad sense for any kind of maps which highlight prediction relevant spatial features in the input data. The `nip` specific implementations via  `GradCAM()` and `GuidedBackprop()` are discussed below.

## Background

A major current research subfield within AI research is the one of Explainable Artificial Intelligence (XAI)  whose aim is to understand the inner workings of machine learning driven predictions. Saliency maps can be regarded as one part of this effort. The question we want to answer is the following: given an output of a network, what aspects of the input data where decisive for the prediction for a specific class. In the case of image data we are interested in image regions which are most relevant for a specific prediction. Note that to first order "relevancy" here is more a statement about the learned representation of the network rather than the input data.

There are several reasons why one might be interested in answering that question

* Improve trust in the prediction: By highlighting regions which are relevant for a prediction we can increase our trust in our model and its ability to generalize. Prediction accuracy might not be a good enough metric to establish that on its own (see next point).
* Detect biases during training: Think of a CNN which differentiates between Wolfs and Huskies. One can imagine a scenario where the network reaches high accuracy but what might be actually happening is that the network looks at the presence of snow for its prediction. Without knowing anything about the inner workings of the network such biases might be difficult to detect when the evaluation of the network is solely based on conventional metrics.
* Human Learning: In a scenario where an AI outperforms humans one can think of gaining deeper understanding of the learned representation as a means of learning from the machines. See for example move 37 in AlphaGo vs Lee Sedol

## Methodology
### Guided Backpropagation
As a first approach one could try to find the most relevant modes in the input data by differentiating the network output for a specific class $`S_c`$ with respect to the input image $`x_0`$
```math
 \frac{\partial S^c(x)}{\partial x_{i,j}}\Bigr\rvert_{x_0}
```
It turns out, however, that this leads to rather noisy results[^1] (see Figure below) which result from interference between positive and negative gradients during backpropagation. This problem is addressed by the Guided Backpropagation[^2] method (`GuidedBackprop()` in `nip`). The basic idea is that negative gradients are suppressed during backpropagation leading to very detailed maps of pixel importance.
<br />![Guided Backpropagation](../images/saliency_backprop.png)<br />
Although both of these methods are based on backpropagation with respect to specific output layer neurons (corresponding to distinct classes) they are in fact *not* class discriminative. The reason for this is rather non-trivial but related to the fact that instance-specific information is encoded in the network during the forward pass[^3] [^4].

### Class activation mapping and Grad-CAM
In order to find a method which is able to differentiate between classes one can meditate on the following two observations

* Deeper layers of CNNs are known to encode higher level semantics
* Spatial correlations are maintained throughout the convolutional part up until the fully connected classifier part.

It could be therefore argued that the output of the last convolutional layer holds most of the semantic and spatial information relevant for prediction. A method based on this train of thought is the Class activation mapping (CAM)[^5] method. The authors look at a CNN where the fully connected part is replaced with a global average pooling layer. Each of the feature map in the final convolutional layer is thus reduced to a single value which in turn is directly coupled to the network output. Since every class has its unique set of weights for this last connection these weights can be interpreted as a importance weighting of the final feature maps. An upscaled, weighted average of the feature maps therefore gives as a class specific activation map.

Although this quite simple approach yields very impressive results (see reference) its shortcomings are rather severe. One has to sacrifice performance (no fully connected part) for interpretability and is restricted to a very specific class of architectures. A generalization (in an actual mathematical sense) of CAM which can be applied to arbitrary networks with a convolutional block without any retraining is a gradient based method called Grad-CAM (Gradient-weighted class activation maps)[^6] (`GradCAM()` in `nip`). The idea is similar to before but here one uses the averaged gradients of the final feature maps $`A^k_{x,y}`$ to compute class $`c`$ specific weights
```math
w^c_k \propto \sum_{i,j}\frac{\partial S^c}{\partial A^k_{i,j}}
```
The class activation map is then given by
```math
M^c(i,j) = \sum_k w^c_kA^k_{i,j}
```
which in turn is put through a ReLU (we are only interested in positive contributions to the class) and then upscaled to input size.
<br />![GradCAM](../images/saliency_gradcam.png)<br />
The method shows very nice class differentiating properties. Note how the network even detects the ear of the fluffy dog behind the cats head as part of him!

We can now combine the Grad-CAM method and the Guided Backpropagation method to generate very detailed, class discriminative maps by combing the outputs of the two methods which is called Guided Grad-CAM (`GuidedGradCAM()` in `nip`)
<br />![GuidedGradCAM](../images/saliency_guidedgradcam.png)<br />

## NIP implementations
### `GuidedBackprop()`
""" Class to generate Guided Backpropagation maps.
    See https://arxiv.org/pdf/1412.6806.pdf

    Attributes:
        model (torch.nn.Module): A CNN model with a convolutional and fully connected section.
            The convolutional part is expected to have ReLUs as their activation function.
            It is assumed that the forward method of the model is equivalent to sequentially
            applying each module in the definition of the convolutional part, flattening the output and
            passing it sequentially through every module in the fully connected part. Consequently,
            activation functions should be  defined via torch.nn layers in the model definition and not
            as torch.nn.functional's in the forward method.
        section_names (list of strings): Denote the names of the convolutional and fully connected sections of the model.
            Default: Detect the names automatically. This will only work if there are exactly two named sections, corresponding
            to the convolutional and fully connected part, in the model defintion.

    Methods:
        __call__(self, x, target_class=[], ignore_lastlayer=False):
            Return network predictions and Guided Backpropagation maps of same dimension as x
    """


`GuidedBackprop()` instances are created by passing CNN models in the form of `torch.nn.Module` subclasses. Furthermore, the layers in `model.ConvBlock` should use ReLUs as their activation function. See for e.g. `helpers.utils_neural_network.Neural_network`
```
Neural_network(
  (ConvBlock): Sequential(
    (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2))
    (1): ReLU(inplace=True)
    (2): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2))
    (3): ReLU(inplace=True)
    (4): Conv2d(32, 16, kernel_size=(3, 3), stride=(2, 2))
    (5): ReLU(inplace=True)
  )
  (Linear_block): Sequential(
    (0): Linear(in_features=400, out_features=100, bias=True)
    (1): ReLU(inplace=True)
    (2): Linear(in_features=100, out_features=1, bias=True)
    (3): Sigmoid()
  )
)
```
#### Methods
##### Call
An instance of `GuidedBackprop()` can then be used to create Guided Backpropagation maps using following parameters
###### Parameters
* `x` (`torch.Tensor`): Batch of input images for which Guided Backprop maps are generated. `x` should have the shape [batch_size, channels, height, width] or [channels, height, width].

###### Optional:
* `target_class` (list of integers): Classes for which Guided Backprop maps are generated. Default is predicted classes.
* `ignore_lastlayer`(bool): If `True`, the network outputs are computed without the last layer in `Linear_block`. Useful if last layer is sigmoid/softmax due do vanishing gradients problem. Default is `False`.

###### Returns
* `model_output` (`torch.Tensor`): `self.model` predictions for `x` and `target_class`. Note: Depending on `ignore_lastlayer` returned output might be before last layer.
* `gbp`(`torch.Tensor`): Guided Backpropagation maps of same dimension as x.

#### Example
```
from helpers.saliency import GuidedBackprop
from helpers.utils_neural_network import Neural_network

# Dummy data
images = torch.rand([4,1,50,50])
# Initialize model
NN = Neural_network([1,50,50])

GBP = GuidedBackprop(NN, mode="2D")
pred, gbp = GBP(images,[0,0,0,0])
```
### `GradCAM()`
    """ Class to generate Gradient-weighted class activation maps.
    See https://arxiv.org/pdf/1610.02391.pdf

    Attributes:
        model (torch.nn.Module): A A CNN model with a convolutional and fully connected section.
            It is assumed that the forward method of the model is equivalent to sequentially
            applying each module in the definition of the convolutional part, flattening the output and
            passing it sequentially through every module in the fully connected part. Consequently,
            activation functions should be  defined via torch.nn layers in the model definition and not
            as torch.nn.functional's in the forward method.
        target_layer (int): Target layer from which features/gradients are extracted for
            CAM generation. Default: Last layer in convolutional section.
        section_names (list of strings): Denote the names of the convolutional and fully connected sections of the model.
            Default: Detect the names automatically. This will only work if there are exactly two named sections, corresponding
            to the convolutional and fully connected part, in the model defintion.
        mode (str): Choose between "2D" and "3D" for two dimensional or three dimensional data respectively.
            Default: "3D"

    Methods:
        __call__(x, target_class=[], ignore_lastlayer = False) :
            Return network predictions and GradCAM maps of same dimension as x
    """
`GradCAM()` instances are created by passing CNN models in the form of `torch.nn.Module` subclasses.
#### Methods
##### Call
An instance of `GradCAM()` can be used to create class activation maps using following parameters
###### Parameters
* `x` (`torch.Tensor`): Batch of input images for which Grad-CAM maps are generated. `x` should have the shape [batch_size, channels, height, width] or [channels, height, width].

###### Optional:
* `target_class` (list of integers): Classes for which maps are generated. Default is predicted classes.
* `ignore_lastlayer`(bool): If `True`, the network outputs are computed without the last layer in `Linear_block`. Useful if last layer is sigmoid/softmax due do vanishing gradients problem. Default is `False`.
###### Returns
* `model_output` (`torch.Tensor`): `self.model` predictions for `x` and `target_class`. Note: Depending on `ignore_lastlayer` returned output might be before last layer.
* `cam`(`torch.Tensor`): GradCAMs for x and target_class(es)

#### Example
```
from helpers.saliency import GradCAM
from helpers.utils_neural_network import Neural_network

# Dummy data
images = torch.rand([4,1,50,50])
# Initialize model
NN = Neural_network([1,50,50])

CAM = GradCAM(NN, mode="2D")
pred, maps  = CAM(images,[0,0,0,0])
```
### `GuidedGradCAM()`
Class to generate guided GradCAM maps. The usage and functionality is identical to `GradCAM()`.

### Utility functions
#### `saliency`
```
    """ Wrapper for GradCAM, Guided Backpropagation and Guided GradCAM methods

    Parameters:
        model (torch.nn.Module): A CNN model with a convolutional and fully connected section.
            It is assumed that the forward method of the model is equivalent to sequentially
            applying each module in the definition of the convolutional part, flattening the output and
            passing it sequentially through every module in the fully connected part. Consequently,
            activation functions should be  defined via torch.nn layers in the model definition and not
            as torch.nn.functional's in the forward method.
            For method = "GuidedBackprop" the convolutional part is expected to have ReLUs as their
            activation function.
        section_names (list of strings): Denote the names of the convolutional and fully connected
            sections of the model. Default: Detect the names automatically. This will only work if
            there are exactly two named sections, corresponding to the convolutional and fully
            connected part, in the model defintion.
        target_layer (int): For method = "GradCAM". Target layer from which features/gradients
            are extracted for CAM generation. Default: Last layer in convolutional part.
        method (str): Choose method from ["GradCAM", "GuidedBackprop", "GuidedGradCAM"]
        mode (str): Choose between "2D" and "3D" for two dimensional or three dimensional data respectively.
            Default:"3D"
"""

```
##### Example
```
from helpers.saliency import saliency
NN = ... # model
CAM = saliency(NN, method = "GradCAM", mode="2D")
preds, cam = CAM(images)
```
#### `saliency_plot`
```
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
```
##### Example
```
from helpers.saliency import saliency, saliency_plot

NN = ... # model

CAM = saliency(NN, method = "GradCAM", mode="2D")
preds, cam = CAM(images)

GBP = saliency(NN, method = "GuidedBackprop", mode="2D")
_ , gbp = GBP(images)

GCAM = saliency(NN, method = "GuidedGradCAM", mode="2D")
_ , gcam = GCAM(images)

# plot batch
saliency_plot(images, mode="2D")

# plot GradCAM as an overlay
saliency_plot(images, cam, mode="2D")

# plot GuidedBackProp
saliency_plot(gbp, mode="2D")

# plot GuidedGradCAM
saliency_plot(gcam)
```
<br />![saliency_plot output](../images/saliency_plot.png)<br />
## References
[^1]: Simonyan et al., *Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps*, 2013. [Arxiv](https://arxiv.org/pdf/1312.6034.pdf)
[^2]: JT Springenberg et al., *Striving for Simplicity: The All Convolutional Net*, 2014. [Arxiv](https://arxiv.org/pdf/1412.6806.pdf)
[^3]: J Gu et al., *Understanding Individual Decisions of CNNs via Contrastive Backpropagation*, 2018. [Arxiv](https://arxiv.org/pdf/1812.02100.pdf)
[^4]: A Mahendran, A Vedaldi *Salient Deconvolutional Networks*, 2016.
[^5]: B Zhou et al., *Learning Deep Features for Discriminative Localization*, 2015. [Arxiv](https://arxiv.org/pdf/1512.04150.pdf)
[^6]: RR Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, 2016. [Arxiv](https://arxiv.org/pdf/1610.02391.pdf)