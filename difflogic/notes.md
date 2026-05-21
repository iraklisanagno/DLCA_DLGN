# Notes on `difflogic`

## 1. Why is there `mnist20x20` in `experiments/main.py`?

`mnist20x20` is MNIST with the black border removed so that a `28x28` image becomes `20x20`.

Code path:
- Dataset selection: `experiments/main.py:44-54`
- Border removal is enabled by `remove_border=args.dataset == 'mnist20x20'` at `experiments/main.py:45-46`
- The actual transform is `MNISTRemoveBorderTransform` in `experiments/mnist_dataset.py:16-60`
- The input dimension is then set to `400` instead of `784` in `experiments/main.py:98-100`

Why they do it:
- The transform trims empty black margins around digits.
- That reduces the input dimensionality from `784` to `400`.
- For logic gate networks, smaller input dimension means fewer possible input pairs and a cheaper model.
- The README examples also treat `mnist20x20` as a separate experiment target: `README.md:187`.

So `mnist20x20` is not a new dataset. It is a cropped MNIST variant intended to remove uninformative border pixels.

## 2. What are `train_set_size` and `valid_set_size` for MNIST?

They define how the original MNIST training split is divided into a training subset and a validation subset.

Code path:
- `experiments/main.py:48-50`

The code does:
- `train_set_size = ceil((1 - args.valid_set_size) * len(train_set))`
- `valid_set_size = len(train_set) - train_set_size`
- `random_split(train_set, [train_set_size, valid_set_size])`

Meaning:
- `args.valid_set_size` is a fraction, defined by `--valid-set-size` / `-vss` at `experiments/main.py:245`
- If `len(train_set) = 60000` and `--valid-set-size 0.1`, then:
  - `train_set_size = ceil(0.9 * 60000) = 54000`
  - `valid_set_size = 6000`

Default behavior:
- The default is `0.0`, so by default there is effectively no validation set.
- In that case the code still creates a split, but the validation subset has length `0`.

## 3. Explain `cifar-10-3-thresholds` and `cifar-10-31-thresholds`. What does the paper mean by color channel resolution? Do I need these transformations? Can I use raw CIFAR-10 instead?

### What the code does

Code path:
- CIFAR dataset selection: `experiments/main.py:55-73`
- Transform definitions: `experiments/main.py:56-63`
- Input dimensions: `experiments/main.py:100-101`

The transforms are:
- `cifar-10-3-thresholds`: `torch.cat([(x > (i + 1) / 4).float() for i in range(3)], dim=0)`
- `cifar-10-31-thresholds`: `torch.cat([(x > (i + 1) / 32).float() for i in range(31)], dim=0)`

`torchvision.transforms.ToTensor()` first maps CIFAR-10 pixels from integers in `[0,255]` to floats in `[0,1]`.
Then the code threshold-binarizes them repeatedly.

For `3-thresholds`, the thresholds are:
- `1/4 = 0.25`
- `2/4 = 0.50`
- `3/4 = 0.75`

For `31-thresholds`, the thresholds are:
- `1/32, 2/32, ..., 31/32`

For each threshold, it creates a binary image saying whether each pixel/channel is above that threshold.
Then it concatenates all of these binary copies along the channel dimension.

So the shapes become:
- Original CIFAR-10 after `ToTensor()`: `3 x 32 x 32`
- `cifar-10-3-thresholds`: `9 x 32 x 32`
- `cifar-10-31-thresholds`: `93 x 32 x 32`

Flattened input dimensions:
- `3 * 32 * 32 * 3 = 9216` for `3-thresholds`
- `3 * 32 * 32 * 31 = 95232` for `31-thresholds`

### What “color channel resolution” means

The paper says the inputs are converted to Boolean values and then processed by logic gates. A logic gate network fundamentally wants binary inputs.
For grayscale MNIST, thresholding to Boolean is easy. For CIFAR-10, each pixel has 3 color channels with many intensity levels, so they need a way to represent color intensity using binary features.

That is what these threshold stacks do: they convert a single real-valued channel into multiple ordered binary indicators.
You can think of this as replacing one analog-like intensity value by a small binary staircase code.

### Concrete per-pixel example

Yes, your interpretation is correct.

If we use `cifar-10-3-thresholds`, then each **single channel** value is represented by 3 threshold bits.
Since each CIFAR-10 pixel has 3 channels `(R, G, B)`, the same pixel becomes:
- `3 bits` for red
- `3 bits` for green
- `3 bits` for blue
- total: `9 bits` for that one pixel

For `cifar-10-31-thresholds`, one pixel becomes:
- `31 bits` for red
- `31 bits` for green
- `31 bits` for blue
- total: `93 bits` for that one pixel

Example pixel:
- suppose one pixel in a CIFAR-10 image has RGB values `(200, 80, 10)` in `[0,255]`
- after `ToTensor()`, this is approximately:
  - `R = 200/255 = 0.784`
  - `G = 80/255 = 0.314`
  - `B = 10/255 = 0.039`

With `3-thresholds`, compare each channel against `0.25`, `0.50`, `0.75`:

Red channel `0.784`:
- `0.784 > 0.25` -> `1`
- `0.784 > 0.50` -> `1`
- `0.784 > 0.75` -> `1`
- red code = `[1, 1, 1]`

Green channel `0.314`:
- `0.314 > 0.25` -> `1`
- `0.314 > 0.50` -> `0`
- `0.314 > 0.75` -> `0`
- green code = `[1, 0, 0]`

Blue channel `0.039`:
- `0.039 > 0.25` -> `0`
- `0.039 > 0.50` -> `0`
- `0.039 > 0.75` -> `0`
- blue code = `[0, 0, 0]`

So that one original RGB pixel becomes the 9-bit vector:
- `[1,1,1, 1,0,0, 0,0,0]`

This is exactly the kind of Boolean-valued representation the logic network wants.

### Concrete whole-image interpretation

A CIFAR-10 image originally has:
- `32 x 32 = 1024` pixels
- `3` channels per pixel
- total raw values: `3072` real-valued channel intensities

After `3-thresholds`:
- each original channel value becomes `3` Boolean values
- so the image has `3072 * 3 = 9216` Boolean features

After `31-thresholds`:
- each original channel value becomes `31` Boolean values
- so the image has `3072 * 31 = 95232` Boolean features

So you can think of the transform as increasing “color resolution” in a binary way:
- `3-thresholds` gives a coarse binary encoding of each color intensity
- `31-thresholds` gives a much finer binary encoding of each color intensity

It is not 8-bit binary encoding in the normal digital-image sense.
It is a threshold encoding: “is this channel above level 1?”, “above level 2?”, ..., “above level n?”

### Why use this for DLGNs?

Because DLGNs are designed around binary logic operations.
The threshold representation converts continuous-valued color inputs into many Boolean-valued features while still preserving intensity ordering.

This is consistent with the paper:
- Figure 1 text says “the pixels of the image are converted into Boolean valued inputs”
- The paper also discusses multiple Boolean outputs per class and bit-count aggregation: extracted paper text, Section 3 and 4

### Do you need these transformations here?

If you want to reproduce the CIFAR experiments in this repo, yes.
The repo’s `experiments/main.py` only supports:
- `cifar-10-3-thresholds`
- `cifar-10-31-thresholds`

It does **not** support raw CIFAR-10 real-valued input in `main.py`.

### Can you use the default CIFAR-10 dataset as it is?

Not in `experiments/main.py` without modifying the code.

Reason:
- The logic-net experiment code assumes Boolean-like inputs for the logic architecture.
- The implemented CIFAR options are specifically the thresholded ones.
- In `experiments/main_baseline.py`, there is a raw-input option called `cifar-10-real-input` at `experiments/main_baseline.py:55, 99, 219`, but that is only for the baseline neural network script, not the DLGN script.

So:
- For the baseline MLP: yes, raw CIFAR exists in this repo
- For DLGN in `experiments/main.py`: no, not without changing the code

## 4. What is `randomly_connected` for `args.architecture`? What other choices do I have?

Code path:
- Architecture option is parsed at `experiments/main.py:249`
- The only implemented branch is `if arch == 'randomly_connected'` at `experiments/main.py:133-142`
- Any other value triggers `raise NotImplementedError(arch)` at `experiments/main.py:146-147`

What it means:
- The network is built as a stack of `LogicLayer`s.
- Each logic neuron receives two fixed inputs.
- The graph structure is sparse and fixed after initialization.
- The architecture is “randomly connected” in the sense used by the paper: the connectivity is initialized pseudo-randomly and then kept fixed.

What other choices exist in the current code:
- None, in practice.
- The CLI parser accepts any string because there is no `choices=` list, but only `randomly_connected` is implemented.

So the real answer is: the current codebase has exactly one architecture option.

## 5. How are the weights initialized? Which method is used? Show where it happens.

Code path:
- `difflogic/difflogic.py:33`

The line is:
```python
self.weights = torch.nn.parameter.Parameter(torch.randn(out_dim, 16, device=device))
```

Meaning:
- Every neuron has 16 trainable logits, one for each possible binary logic operator.
- They are initialized from a standard normal distribution via `torch.randn(...)`.

This matches the paper:
- Section 4.1 says: “For the initial parameterization of each neuron, we draw elements of `w` independently from a standard normal distribution.”

During training these logits are converted to a probability distribution by softmax:
- `difflogic/difflogic.py:123` in CUDA training path
- `difflogic/difflogic.py:104` in Python training path

So the initialization method is standard normal initialization of the 16 operator logits per neuron.

## 6. What is `packbits_eval`? Is it mentioned in the paper?

Code path:
- Function definition: `experiments/main.py:198-210`
- CLI flag: `experiments/main.py:238-239`
- It wraps the input as `PackBitsTensor(...)` at `experiments/main.py:204`

What it does:
- It switches the model to eval mode.
- It rounds the input, converts it to bool, packs bits across the batch dimension, and runs the `PackBitsTensor` CUDA inference path.
- This is an additional evaluation path intended to test the packed-bit GPU inference mode.

What `PackBitsTensor` is:
- It is a GPU bit-packing inference representation: `README.md:114-133`
- The layer handles it specially in `difflogic/difflogic.py:75-88, 134-149`

Is it mentioned in the paper?
- Not under the name `PackBitsTensor`.
- The README explicitly says the inference modes in the released library differ somewhat from the paper-era implementation: `README.md:166-169`
- So `packbits_eval` is a library/repo feature for evaluating the optimized CUDA inference path, not a core paper training concept.

One code detail:
- At `experiments/main.py:315`, `valid_acc_eval` mistakenly uses `train_loader` instead of `validation_loader`.
- So that line is likely a small bug in the script.
- FIXED by IRAKLIS

## 7. What is `compile_model`?

Code path:
- CLI flag: `experiments/main.py:240`
- Execution block: `experiments/main.py:335-379`
- Implementation class: `difflogic/compiled_model.py`
- Overview in README: `README.md:137-164`

What it does:
- After training, it converts the learned logic network into generated C code and compiles it into a shared library.
- That compiled model is then evaluated on CPU.

In `main.py`, when `--compile_model` is set:
- it creates `CompiledLogicNet(...)`
- compiles it with `gcc`
- saves a shared object under `lib/...so`
- flattens the test data to boolean numpy arrays
- runs the compiled CPU model and reports accuracy

So `compile_model` is not about training. It is a post-training export/compile step for very fast CPU inference.

### Why would you want it in C at all?

Because the final trained DLGN is fundamentally a fixed network of discrete logic gates.
Once training is over, you no longer need:
- autograd
- softmax over gates during training
- Python object dispatch per layer
- general PyTorch tensor machinery for every operation

At inference time, the model can be reduced to a static logic circuit.
That is exactly the case where compiled C is attractive.

The paper’s whole motivation is that logic-gate networks are especially good at **fast inference**, not just trainability.
The paper explicitly emphasizes that after discretization the network can be executed very efficiently with bitwise logic operations.

### Why not just do regular Python inference like in normal DNNs?

You can. There are already two non-compiled inference routes in this repo:
- regular PyTorch-style inference with the model in `.eval()` mode
- the CUDA `PackBitsTensor` inference path

So `compile_model` is optional, not required.

But regular Python/PyTorch inference has overheads that matter here:
- Python function-call overhead
- PyTorch module/tensor dispatch overhead
- more generic kernels than necessary
- less direct control over bit-packing and bitwise execution on CPU

For ordinary DNNs, this overhead is often acceptable because the heavy work is large matrix multiplies or convolutions done by optimized BLAS/CUDA libraries.
For a DLGN, the final model is much closer to a big sparse Boolean circuit. In that setting, the framework overhead can become a much larger fraction of total runtime.

So the compiled C path exists to remove that overhead and exploit the fact that the final network is just logic.

### When should you use it?

Use `--compile_model` if:
- you care about very fast CPU inference
- you want a standalone compiled artifact
- you want to benchmark the “hard” discretized model as a logic program rather than as a PyTorch graph

Do not use it if:
- you are still training
- you are just debugging the model
- normal PyTorch inference speed is already sufficient
- you want the simplest workflow

### Practical summary

There are three levels here:
- training / ordinary evaluation in PyTorch: easiest, most flexible
- `PackBitsTensor` CUDA eval: optimized GPU inference path
- `CompiledLogicNet` / `compile_model`: optimized CPU inference/export path

So the answer is not “you must have it in C.”
The answer is: they added the C path because DLGNs become especially efficient when turned into a fixed compiled logic program, and that is one of the main points of the method.

## 8. What is the option `connections`?

Code path:
- CLI flag: `experiments/main.py:248`
- Passed into each layer at `experiments/main.py:120`
- Used in `LogicLayer.__init__` at `difflogic/difflogic.py:52-55`
- Implemented in `get_connections(...)` at `difflogic/difflogic.py:154-169`

Choices:
- `random`
- `unique`

What they mean:

### `random`
Implemented at `difflogic/difflogic.py:158-165`
- It pseudo-randomly assigns two input indices to each output neuron.
- Inputs may repeat across neurons.
- This matches the paper’s general idea of fixed random sparse connectivity.

### `unique`
Implemented via `get_unique_connections(...)` in `difflogic/functional.py:76-124`
- It tries to systematically cover input pairs so all inputs are used.
- It starts with adjacent pairs like `(0,1), (2,3), ...`
- then offset pairs like `(1,2), (3,4), ...`
- then larger offsets if needed
- then shuffles the resulting pair list

So `unique` is still randomized at the final permutation stage, but it is much more structured than `random` and tries to cover the input space better.

## 9. What is `grad-factor`?

Code path:
- CLI flag: `experiments/main.py:253`
- Passed into layers at `experiments/main.py:120`
- Stored in layer at `difflogic/difflogic.py:37`
- Applied in forward at `difflogic/difflogic.py:82-83`
- Implemented by custom autograd op `GradFactor` in `difflogic/functional.py:130-138`

What it does:
- In the forward pass, it does nothing.
- In the backward pass, it multiplies the incoming gradient by `grad_factor`.

The implementation is:
```python
class GradFactor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, f):
        ctx.f = f
        return x

    @staticmethod
    def backward(ctx, grad_y):
        return grad_y * ctx.f, None
```

So `grad-factor` is a gradient scaling trick.

Why it exists:
- The code comments and README say that for deep models, gradients can vanish.
- Scaling the backward signal can help training deeper logic networks.

This is not changing the function computed by the network. It is changing the gradient magnitude used during optimization.

## 10. Where do I define how many outputs per class the DLGN will have?

This is determined indirectly by:
- total output neurons in the last `LogicLayer`: `args.num_neurons` / `-k`
- number of classes from the dataset
- aggregation by `GroupSum(class_count, args.tau)`

Code path:
- Number of output neurons per layer: `experiments/main.py:128, 135-137, 250`
- Number of classes: `experiments/main.py:123, 105-116`
- Aggregation layer: `experiments/main.py:139-142`
- Grouping logic: `difflogic/difflogic.py:195-196`

The effective outputs per class are:
- `num_neurons / class_count`

Example:
- MNIST has `class_count = 10`
- if you set `-k 64000`, then each class gets `6400` output neurons before aggregation
- `GroupSum` reshapes the last dimension into `k` groups and sums each group

Important constraint:
- `GroupSum` requires divisibility: `x.shape[-1] % self.k == 0` at `difflogic/difflogic.py:195`
- So the total number of output neurons must be divisible by the number of classes

This is also exactly the paper’s idea:
- multiple output neurons per class are aggregated by summation / bit-counting to obtain class scores
- see the extracted paper text in Sections 3 and 4
