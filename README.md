# ViewFormer
Inspect and visualize Transformer's characteristics.

This library includes a set of tools to capture and visualize the weights and activations of a model.

### How to use:
Drop the whole folder into your project and import the module. 

Below are some examples to obtain the weights of a layer:

```python
from viewFormer.utils import get_model_layers, get_layer_weights

# Obtain all the layers of a model: (returns a list of tuples with the layer name and the layer object)
all_layers = get_model_layers(model)

# Obtain specific layers of a model: (returns a list of tuples with the layer name and the layer object)
layer_names = [f'blocks.0.attn.qkv', f'blocks.0.mlp.fc1', f'blocks.0.mlp.fc2']
block0_layers = get_model_layers(model, match_names=layer_names, match_types=['Linear'])

# To get the weights of a layer:
block0_qkv = block0_layers[0][1]
weight = block0_qkv.weight # get weight parameter
weights = get_layer_weights(layer) # or use get_layer_weights function, if there are many paramameters in a layer (returns a list of tuples with the parameter name and the parameter tensor)
```


To get the activations of a layer, we provide a HookHandler class that wraps around pytorch hooks, which makes recording layer outputs simple:

```python
from viewFormer.hooks import HookHandler, get_act_func, get_avg_act_func
from viewFormer.data import calibrate

# initialize HookHandler and the dictionary to store the outputs
handler = HookHandler()
layer_outputs = {}

# get the layers you want to record
layer_names = [f'blocks.0.attn.qkv', f'blocks.0.mlp.fc1', f'blocks.0.mlp.fc2']
block0_layers = get_model_layers(model, match_names=layer_names, match_types=['Linear'])

# create hooks for the layers (<layers>, <hook function>, <dictionary to store the outputs>)
handler.create_hooks(block0_layers, get_act_func, layer_outputs)

# calibrate the model and record the outputs
with torch.autocast(device_type="cuda"):
    calibrate(model, list(test_loader)[:256])

# remove the hooks
handler.remove_hooks()
```

Check out ```example.ipynb```, for the full code of visualizing weight/activation values of your model.
