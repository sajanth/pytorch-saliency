"""
Methods for generating gradient-based class activation maps (GradCAM) and
guided backpropagation maps.
"""
import torch

def saliency(model, section_names=None, target_layer=None, method=None, mode="3D", adaptive_layer=None):
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
        target_layer (list of strings): Target layer from which features/gradients are extracted for
            CAM generation. The list can either contain just the module name or the module name as the first and
            the submodule name as the second entry. To see an overview of all the modules and submodules run `print(model)`.
            If a module contains submodules and only the module name is given, then the last submodule in
            the given module is selected as target layer. Example: In Resnet from torchvision.models we would set target_layer
            = ["layer3", "1"] if we would like to use the feature maps of the second BasicBlock in layer 3 for the generation of the CAM.
            Default: Last layer in convolutional section (i.e. in section_names[0]).
        method (str): Choose method from ["GradCAM", "GuidedBackprop", "GuidedGradCAM"]
        mode (str): Choose between "2D" and "3D" for two dimensional or three dimensional data respectively.
            Default:"3D"
        adaptive_layer (string): Module which should be applied between the convolutional and fully connected part before flattening.
            Example: Some models `torchvision.models utilize `AdaptiveAvgPool2D` to handle varying input image sizes. In standard torch
            Resnet implementation, for example, the module is named "avgpool".
            Default: None

    """
    methods = ["GradCAM", "GuidedBackprop", "GuidedGradCAM"]
    assert method in methods, "{} is not a valid method name. Choose one of {}".format(method, methods)

    if method == "GradCAM":
        return GradCAM(model=model, target_layer=target_layer, section_names=section_names, mode=mode, adaptive_layer=adaptive_layer)
    elif method == "GuidedBackprop":
        return GuidedBackprop(model=model, section_names=section_names)
    elif method == "GuidedGradCAM":
        return GuidedGradCAM(model=model, target_layer=target_layer, section_names=section_names, mode=mode, adaptive_layer=adaptive_layer)

class GuidedBackprop():
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
    def __init__(self, model, section_names=None):
        if section_names is None:
            section_names = [name for name, _ in model._modules.items()]

            assert len(section_names) == 2, ("Model does not have two separate sections. Instead has: {}."
                                             " Modules in model defintion must be grouped into a convolutional and fully"
                                             " connected section e.g. 'Conv_block' and 'Linear_block'. You can specify these"
                                             " sections also manually via the 'section_names' argument".format(section_names))
        else:
            names = [name for name, _ in model._modules.items()]
            assert section_names[0] in names, "{} section not found in model".format(section_names[0])
            assert section_names[1] in names, "{} section not found in model".format(section_names[1])

        self.conv_name = section_names[0]
        self.classifier_name = section_names[1]
        self.model = model
        self.model.eval()
        self._grads = None
        # temp store forward outputs of network, each entry i corresponds to forward outputs at layer i
        self._forward_outputs = []
        # hook input layer in order to save guided backprop map
        self._hook_input_layer()
        # hook relu layers with modified gradients
        self._mod_relus()

    def _save_grads(self, module, grad_in, grad_out):
        self._grads = grad_in[0]

    def _hook_input_layer(self):
        # save gradients of first layer during backprop
        first_layer = list(getattr(self.model, self.conv_name)._modules.items())[0][1]
        first_layer.register_backward_hook(self._save_grads)

    def _mod_relus(self):
        """
        Modify relu layers for forward and backward pass
        """
        # gradient values through relu are set to zero if gradient value is negative during backprop
        def backward_mod(module, grad_in, grad_out):
            forward_output = self._forward_outputs[-1]
            forward_output[forward_output > 0] = 1 #derivative of relu for x>0 is 1
            # the gradient at i is the local gradient times the incoming gradient from layer i+1
            # set these incoming gradients to 0 where negative
            mod_grad = forward_output * torch.max(grad_in[0], torch.zeros(grad_in[0].shape))
            del self._forward_outputs[-1]
            return (mod_grad,)

        # save outputs at each relu layer in _forward_outputs such that we can compute local gradients by hand
        def forward_mod(module, inp, out):
            self._forward_outputs.append(out)

        for _, module in getattr(self.model, self.conv_name)._modules.items():
            if isinstance(module, torch.nn.modules.activation.ReLU):
                module.register_backward_hook(backward_mod)
                module.register_forward_hook(forward_mod)

    def _forward(self, x, ignore_lastlayer):
        """
        Move through the net and return model output x
        (before final softmax/sigmoid if ignore_lastlayer = True)
        """
        for name, module in getattr(self.model, self.conv_name)._modules.items():
            x = module(x)
        x_shape = x.shape
        x = x.view([x_shape[0], 1, -1])
        for name, module in getattr(self.model, self.classifier_name)._modules.items():
            # skip last layer if ignore_lastlayer = True
            if int(name) != len(getattr(self.model, self.classifier_name)._modules.items())-int(ignore_lastlayer):
                x = module(x)
        return x

    def __call__(self, x, target_class=None, ignore_lastlayer=False):
        """ Generate Guided Backpropagation maps for input x.

        Parameters:
            x (torch.Tensor): Batch of input images for which Guided Backprop maps
                are generated. Shape should be [batch_size, channels, height, width]
                or [channels, height, width]
            target_class (list of integers): Classes for which Guided Backprop maps
                are generated. Default is predicted classes.
            ignore_lastlayer: If True, the network outputs are computed without the last
                layer in the fully connected section. Useful if last layer is sigmoid/softmax due do
                vanishing gradients problem. Default is False.

        Returns:
            gbp(torch.Tensor): Guided Backpropagation maps of same dimension as x.
        """

        # check if x holds gradients, if not activate for the duration of function call
        x_grad = x.requires_grad
        if not x_grad:
            x.requires_grad = True

        model_output = self._forward(x, ignore_lastlayer)
        self.model.zero_grad()

        # vector for gradient computation singling out the targeted classes
        one_hot_class = torch.zeros(x.shape[0], model_output.shape[-1])
        if target_class is not None:
            target_class = [target_class] if isinstance(target_class, int) else target_class
        else:
            target_class = []
            for _, out in enumerate(model_output):
                target_class.append(int(torch.argmax(out, -1)))
            print("Computing GuidedBackprop for classes ", target_class)
        for i, categ in enumerate(target_class):
            one_hot_class[i][categ] = 1
        one_hot_class = one_hot_class.unsqueeze(1)

        model_output.backward(gradient=one_hot_class, retain_graph=True)
        gbp = self._grads
        if not x_grad:
            x.requires_grad = False
        return model_output, gbp

class _GradCamExtract():
    """Support class for the internal use of GradCAM().

    Take model with a convolutional and fully connected section and outputs features maps of a
    targeted layer (default: last layer of convolutional section) for a input x as well as overall
    network output. Furthermore, the targeted layer is hooked such that during backprop
    the corresponding gradients are saved in self.grads.
    """
    def __init__(self, model, target_layer, conv_layers, classifier_layers, adaptive_layer):
        self.model = model
        self.conv_layers = conv_layers
        self.classifier_layers = classifier_layers
        self.adaptive_layer = adaptive_layer

        if target_layer is None:
            target_layer_names = [names for names, _ in getattr(self.model, conv_layers[-1])._modules.items()]
            self.target_layer = [conv_layers[-1], target_layer_names[-1]] if target_layer_names else [conv_layers[-1], None]
        else:
            self.target_layer = target_layer
        self._grads = None

    def _save_grads(self, grad):
        self._grads = grad

    def __call__(self, x):
        """
        Move through the net, hook at targeted layer and return
        target activation and model output x.
        """
        target_output = None
        for layer in self.conv_layers:
            if not getattr(self.model, layer)._modules.items():
                x = getattr(self.model, layer)(x)
                if layer == self.target_layer[0] and self.target_layer[1] is None:
                    x.register_hook(self._save_grads)
                    target_output = x
            else:
                for name, module in getattr(self.model, layer)._modules.items():
                    x = module(x)
                    if layer == self.target_layer[0] and name == self.target_layer[1]:
                        x.register_hook(self._save_grads)
                        target_output = x

        if self.adaptive_layer is not None:
            x = getattr(self.model, self.adaptive_layer)(x)
        x_shape = x.shape
        x = torch.reshape(x, (len(x), 1, -1))

        for layer in self.classifier_layers:
            if not getattr(self.model, layer)._modules.items():
                x = getattr(self.model, layer)(x)
            else:
                for name, module in getattr(self.model, layer)._modules.items():
                    x = module(x)

        return target_output, x

class GradCAM():
    """ Class to generate Gradient-weighted class activation maps.
    See https://arxiv.org/pdf/1610.02391.pdf

    Attributes:
        model (torch.nn.Module): A 'torchvision.models'-like CNN model with convolutional and fully connected sections.
            It is assumed that the forward method of the model is equivalent to sequentially
            applying each module in the definition of the convolutional part, flattening the output and
            passing it sequentially through every module in the fully connected part. Notably,
            activation functions should be defined via torch.nn layers in the model definition and not
            as torch.nn.functional's in the forward method.
        section_names (list of list of strings): Denote the names of the modules which define the convolutional (first entry) and fully connected
            (second entry) sections of the model. All modules which define the forward pass of the model
            should be sequentally listed here (exception: special transformations between the two sections like adaptive pooling; use the
            `adaptive_layer` option for that). The flattening between the convolutional part and fully connected part is handled automatically.
            To see an overview of all the sections/modules and submodules run `print(model)`.
            Examples: For Resnet18 from torchvision.models we would have section_names=
            [["conv1","bn1","relu","maxpool","layer1","layer2","layer3","layer4"],["fc"]].
            For VGG16 from torchvision.models
            we have section_names=[["features"], ["classifier"]].
            Default: Detect the names automatically. This will only work if there are exactly two named sections, corresponding
            to the convolutional and fully connected part, in the model defintion.
        target_layer (list of strings): Target layer from which features/gradients are extracted for
            CAM generation. The list can either contain just the module name or the module name as the first and
            the submodule name as the second entry. To see an overview of all the modules and submodules run `print(model)`.
            If a module contains submodules and only the module name is given, then the last submodule in
            the given module is selected as target layer. Example: In Resnet from torchvision.models we would set target_layer
            = ["layer3", "1"] if we would like to use the feature maps of the second BasicBlock in layer 3 for the generation of the CAM.
            Default: Last layer in convolutional section (i.e. in section_names[0]).
        adaptive_layer (string): Module which should be applied between the convolutional and fully connected part before flattening.
            Example: Some models `torchvision.models utilize `AdaptiveAvgPool2D` to handle varying input image sizes.
            Default: None
        mode (str): Choose between "2D" and "3D" for two dimensional or three dimensional data respectively.
            Default: "3D"

    Methods:
        __call__(x, target_class=[]) :
            Return network predictions and GradCAM maps of same dimension as x

    Example:
        resnet = model.resnet18(pretrained=True)
        resnet.eval()

        cam = GradCAM(
            resnet,
            section_names=[["conv1","bn1","relu","maxpool","layer1","layer2","layer3","layer4"],["fc"]],
            adaptive_layer="avgpool",
            mode="2D"
        )

        pred, maps = cam(img, target_class=282)

    """

    def __init__(self, model, section_names=None, target_layer=None, adaptive_layer=None, mode="3D"):

        self.model = model
        self.model.eval()

        if section_names is None:
            section_names = [[name] for name, _ in model._modules.items()]

            assert len(section_names) == 2, ("Model does not have two separate sections. Instead has: {}."
                                             " Modules in model defintion must be grouped into a convolutional and fully"
                                             " connected section. You can specify these"
                                             " sections also manually via the 'section_names' argument".format(section_names))
        else:
            assert len(section_names) == 2, ("`section_names` needs to be a list of two lists, e.g. [['features'],['classifier']]")

            section_names = [[section_names[0]], [section_names[1]]] if isinstance(section_names[0], str) and isinstance(section_names[1], str) else section_names
            assert isinstance(section_names[0], list), ("`section_names` needs to be a list of two lists, e.g. [['features'],['classifier']]")
            assert isinstance(section_names[1], list), ("`section_names` needs to be a list of two lists, e.g. [['features'],['classifier']]")

            names = [name for name, _ in model._modules.items()]
            check_conv = [section in names for section in section_names[0]]
            check_fc = [section in names for section in section_names[1]]
            assert all(check_conv), "{} section(s) not found in model".format([section_names[0][j] for j in [i for i, x in enumerate(check_conv) if not x]])
            assert all(check_fc), "{} section(s) not found in model".format([section_names[1][j] for j in [i for i, x in enumerate(check_fc) if not x]])

        if target_layer is not None:
            target_layer = [target_layer] if isinstance(target_layer, str) else target_layer
            assert len(target_layer) == 2 or len(target_layer) == 1, "`target_layer` needs to be a list with either one or two string elements denoting the name of the modul and submodule respectively, e.g. ['features', '1']."
            assert all(isinstance(layer_name, str) for layer_name in target_layer), "`target_layer` is not a list of strings."

            if len(target_layer) == 1:
                if not getattr(self.model, target_layer[0])._modules.items():
                    target_layer.append(None)
                else:
                    target_layer.append([names for names, _ in getattr(self.model, target_layer[0])._modules.items()][-1])

        self._extractor = _GradCamExtract(self.model, target_layer, conv_layers=section_names[0], classifier_layers=section_names[1], adaptive_layer=adaptive_layer)

        assert mode in ("2D", "3D"), f"{mode} is not a valid mode. Must be '2D' or '3D'."
        self.mode = mode

    def __call__(self, x, target_class=None):
        """ Generate class activation maps for input x.

        Parameters:
            x (torch.Tensor): Batch of input images for which CAMs should be generated
            target_class (list of integers): Classes for which CAMs should be generated.
                Default is predicted classes.

        Returns:
            model_output(torch.Tensor): self.model predictions for x and target_class
            cam(torch.Tensor): GradCAMs for x and target_class(es)
        """
        if self.mode == "2D":
            assert len(x.shape) == 4, "Input for 2D images must be of form [batch_size, channels, height, width]. For 3D CNN use mode='3D.'"
        else:
            assert len(x.shape) == 5, "Input for 3D images must be of form [batch_size, channels, depth, height, width]. For 2D CNN use mode='2D'."

        # pass x through network and compute activation maps of targeted layer
        target_output, model_output = self._extractor(x)

        # vector(matrix) for gradient computation singling out the target class(es)
        one_hot_class = torch.zeros(x.shape[0], model_output.shape[-1])
        if target_class is not None:
            target_class = [target_class] if isinstance(target_class, int) else target_class
        else:
            # use predicted class if target class not provided
            target_class = []
            for _, out in enumerate(model_output):
                target_class.append(int(torch.argmax(out, -1)))
            print("Computing CAMs for classes ", target_class)
        for i, categ in enumerate(target_class):
            one_hot_class[i][categ] = 1
        one_hot_class = one_hot_class.unsqueeze(1)

        self.model.zero_grad()
        model_output.backward(gradient=one_hot_class, retain_graph=True)
        target_gradients = self._extractor._grads

        # compute weights for target outputs
        if self.mode == "2D":
            alpha = target_gradients.mean((-2, -1))
            cam = torch.zeros(target_output[:, 0, :, :].shape)
        else:
            # handle case where the target layer outputs are 2D
            if len(target_gradients.shape) != 5:
                target_gradients = target_gradients.unsqueeze(2)

            alpha = target_gradients.mean((-3, -2, -1))
            cam = torch.zeros(target_output[:, 0, :, :, :].shape)

        # assign weight to each feature map and combine
        for j in range(alpha.shape[0]):
            for i, weight in enumerate(alpha[j]):
                cam[j] += weight*target_output[j, i]

        cam = torch.max(cam, torch.zeros(cam.shape))
        for i in range(cam.shape[0]):
            cam[i] = (cam[i] - torch.min(cam[i]))/(torch.max(cam[i])-torch.min(cam[i]))

        if self.mode == "2D":
            cam = torch.nn.functional.interpolate(cam.unsqueeze(1), x.shape[-2:], mode="bilinear")
        else:
            cam = torch.nn.functional.interpolate(cam.unsqueeze(1), x.shape[-3:], mode="trilinear")

        return model_output, cam

class GuidedGradCAM():
    """ Class to generate Guided Gradient-weighted class activation maps.
    See https://arxiv.org/pdf/1610.02391.pdf

    Attributes:
        model (torch.nn.Module): A CNN model with a convolutional and fully connected section.
            The convolutional part is expected to have ReLUs as their activation function.
            It is assumed that the forward method of the model is equivalent to sequentially
            applying each module in the definition of the convolutional part, flattening the output and
            passing it sequentially through every module in the fully connected part. Consequently,
            activation functions should be  defined via torch.nn layers in the model definition and not
            as torch.nn.functional's in the forward method.
        target_layer (list of strings): Target layer from which features/gradients are extracted for
            CAM generation. The list can either contain just the module name or the module name as the first and
            the submodule name as the second entry. To see an overview of all the modules and submodules run `print(model)`.
            If a module contains submodules and only the module name is given, then the last submodule in
            the given module is selected as target layer. Example: In Resnet from torchvision.models we would set target_layer
            = ["layer3", "1"] if we would like to use the feature maps of the second BasicBlock in layer 3 for the generation of the CAM.
            Default: Last layer in convolutional section (i.e. in section_names[0]).
        section_names (list of strings): Denote the names of the convolutional and fully connected
            sections of the model. Default: Detect the names automatically. This will only work if there
            are exactly two named sections, corresponding to the convolutional and fully connected part,
            in the model defintion.
        mode (str): Choose between "2D" and "3D" for two dimensional or three dimensional data respectively.
            Default: "3D"

    Methods:
        __call__(x, target_class=[], ignore_lastlayer = False) :
            Return network predictions and GuidedGradCAM maps of same dimension as x
    """
    def __init__(self, model, target_layer=None, section_names=None, mode="3D", adaptive_layer=None):
        if section_names is None:
            section_names = [name for name, _ in model._modules.items()]
            assert len(section_names) == 2, ("Model does not have two separate sections. Instead has: {}."
                                             " Modules in model defintion must be grouped into a convolutional and fully"
                                             " connected section e.g. 'Conv_block' and 'Linear_block'. You can specify these"
                                             " sections also manually via the 'section_names' argument".format(section_names))
        else:
            names = [name for name, _ in model._modules.items()]
            assert section_names[0] in names, "{} section not found in model".format(section_names[0])
            assert section_names[1] in names, "{} section not found in model".format(section_names[1])

        self.conv_name = section_names[0]
        self.classifier_name = section_names[1]
        self.model = model
        self.model.eval()
        self.adaptive_layer= adaptive_layer

        # if target layer is not provided, use last layer of ConvBlock
        if target_layer is None:
            self.target_layer = [self.conv_name, str(len(getattr(self.model, self.conv_name)._modules.items())-1)]
        else:
            self.target_layer = target_layer

        assert mode in ("2D", "3D"), f"{mode} is not a valid mode. Must be '2D' or '3D'."
        self.mode = mode

    def __call__(self, x, target_class=None):
        """ Generate class activation maps for input x.
        Parameters:
            x (torch.Tensor): Batch of input images for which CAMs should be generated
            target_class (list of integers): Classes for which CAMs should be generated.
                Default is predicted classes.

        Returns:
            model_output(torch.Tensor): self.model predictions for x and target_class(es)
            cam(torch.Tensor): GuidedGradCAMs for x and target_class(es)
            """
        CAM = GradCAM(model=self.model, target_layer=self.target_layer, section_names=[self.conv_name, self.classifier_name], mode=self.mode, adaptive_layer=self.adaptive_layer)
        GBP = GuidedBackprop(self.model, [self.conv_name, self.classifier_name])
        pred, maps = CAM(x, target_class)
        _, gbp = GBP(x, target_class)
        return pred, gbp*maps
