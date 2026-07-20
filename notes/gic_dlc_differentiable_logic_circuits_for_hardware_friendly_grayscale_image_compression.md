### GIC-DLC: Differentiable Logic Circuits for Hardware-Friendly Grayscale Image Compression - Summary notes


In this paper, the authors apply the broader **Differentiable Logic Circuit** idea to **lossless grayscale image compression**.

Their contribution is not a new DLGN parameterization such as IWP. Instead, they ask:

> Can learned image compression retain the compression benefits of neural codecs while replacing expensive floating-point networks with LUT-based logic circuits?

They propose **GIC-DLC: Grayscale Image Compression with Differentiable Logic Circuits**. 

# 1. What problem are they solving?

Traditional codecs such as PNG, WebP, and JPEG-XL are:

* fast,
* relatively energy-efficient,
* often supported by dedicated hardware,

but they are hand-designed and may not exploit a specific image distribution optimally.

Learned codecs can predict image statistics more accurately and achieve better compression, but typically require:

* convolutional networks or MLPs,
* floating-point arithmetic,
* many multiply-accumulate operations,
* substantial energy and latency.

The authors try to combine:

```text
Learned compression quality
+
hardware-friendly LUT inference
```

Their target is especially edge hardware such as cameras, phones, and drones. 

# 2. This is lossless compression

The reconstructed image is exactly the same as the input.

The core idea is not to directly output compressed bits. Instead, the model predicts a probability distribution for every pixel.

For example, for a pixel (x), the model may predict:

```text
P(x = 0)   = 0.001
P(x = 1)   = 0.002
...
P(x = 142) = 0.30
P(x = 143) = 0.42
...
```

An entropy coder then assigns:

* fewer bits to likely values,
* more bits to unlikely values.

They use **Asymmetric Numeral Systems**, or ANS, for the final entropy coding.

Thus:

```text
Logic model
    ↓
Pixel probability distribution
    ↓
ANS entropy coder
    ↓
Compressed bitstream
```

The better the probability prediction, the fewer bits are needed.

# 3. Hierarchical multi-resolution compression

The authors compress the image at multiple resolutions.

Starting from the original image (x^{(0)}), they repeatedly apply (2\times2) average pooling:

$
x^{(\ell)}
==========

\operatorname{avgpool}_2(x^{(\ell-1)}).
$

With two downsampling levels, the model has:

```text
Original resolution
      ↓
Half resolution
      ↓
Quarter resolution
```

Decoding starts from the coarsest image and progressively reconstructs finer-resolution images.

The diagram on page 2 shows this coarse-to-fine decoding process. 

The motivation is:

* coarse levels capture global shape,
* finer levels capture details,
* the coarse reconstruction provides a strong prediction for the next level.

# 4. Two models at every resolution

At each scale, the authors use two models.

## Upsampling model, UPS

The UPS predicts a (2\times2) block of higher-resolution pixels from a local neighborhood in the lower-resolution image.

Conceptually:

```text
Coarse neighborhood
       ↓
UPS logic network
       ↓
Predicted 2 × 2 pixel block
```

It produces a point estimate:

$
\tilde{x}^{(i)}.
$

This is similar to learned super-resolution. It gives the decoder an initial estimate of the high-resolution image.

The UPS is trained using mean-squared error.

## Autoregressive model, ARM

The ARM predicts a probability distribution for the current pixel using:

* already decoded neighboring pixels,
* the UPS estimate for pixels not yet decoded.

It predicts the parameters of a Laplace distribution:

$
\mu,\sigma.
$

The distribution is converted into discrete probabilities over pixel values (0,\ldots,255), and the true pixel is entropy-coded using those probabilities.

Thus:

```text
UPS:
predict what the pixel probably is

ARM:
predict how uncertain that prediction is
and assign probabilities to all values
```

Both UPS and ARM are implemented using differentiable logic circuits. 

# 5. They do not use ordinary 2-input DLGN gates

This is an important distinction.

The authors use **NeuraLUT**, not the original DLGN gate formulation.

Each node is effectively a **6-input lookup table**.

For six binary inputs, there are:

$
2^6=64
$

possible input patterns.

After training, they evaluate all 64 combinations and store the node as a LUT.

Inference becomes:

```text
Six input bits
      ↓
6-bit LUT address
      ↓
Stored output bit
```

This is much closer to FPGA LUT-based computation and weightless neural networks than the original 2-input DLGN neuron.

# 6. How is the LUT learned?

During training, the LUT is represented by a small neural network.

That neural network receives the six binary inputs and produces a soft output in ([0,1]).

They add logistic noise before the final sigmoid and anneal a node-temperature parameter so that outputs gradually become binary.

After training:

1. Evaluate all 64 possible inputs.
2. Convert their outputs to binary values.
3. Store the resulting 64-entry truth table.
4. Discard the training neural network.

Therefore:

```text
Training:
small neural network approximates the LUT

Inference:
only the 64-entry Boolean LUT remains
```

This is why the paper calls the system a differentiable logic circuit, despite using a neural network internally during training.

# 7. Learnable connectivity

The connections feeding each LUT are also learned.

For every node input, the model learns a softmax distribution over candidate signals.

During training:

$
a=\sum_j p_j x_j.
$

The connection temperature is gradually reduced so the distribution becomes sharper.

After training:

$
j^*=\arg\max_j p_j
$

selects one fixed connection.

So the final model learns both:

* the six inputs connected to each LUT,
* the 64-bit truth table implemented by that LUT.

This is closer to the learnable-connectivity papers we discussed than to the original fixed-random DLGN.

# 8. How do they process 8-bit grayscale pixels?

Logic circuits need binary inputs, but grayscale pixels range from 0 to 255.

The authors use **thermometer encoding** with 255 thresholds.

For a pixel value (v), they create indicators such as:

$
[v>0], [v>1], [v>2],\ldots,[v>254].
$

For example, if:

$
v=3,
$

then the thermometer code begins approximately as:

$
[1,1,1,0,0,\ldots].
$

This provides a binary, order-preserving representation of grayscale intensity.

It is simple and logic-friendly, but creates a very large input representation.

# 9. Output representation

The final layer contains many binary outputs.

They average these bits to produce a continuous value:

$
y=\frac{1}{M}\sum_{j=1}^{M}b_j.
$

For the predicted mean (\mu), they scale this average to:

$
[0,255].
$

For (\sigma), they apply another transformation to obtain a wider positive range.

This resembles Group-Sum, but instead of using the sum to select a class, they use the average to approximate a continuous numerical value.

That is an important extension of logic networks beyond classification.

# 10. Experimental setup

They train on EMNIST ByClass, which contains:

* handwritten digits,
* uppercase letters,
* lowercase letters.

They also evaluate out of distribution on:

* KMNIST,
* Fashion-MNIST.

Both ARM and UPS use:

* two layers,
* 1,024 LUTs per layer,
* 6-input LUT nodes,
* local kernel size 5,
* two resolution levels.



# 11. Compression results

They report actual coded bits per pixel.

On EMNIST:

| Codec             | Bits per pixel |
| ----------------- | -------------: |
| PNG               |           4.18 |
| WebP              |           3.34 |
| JPEG-XL           |           3.20 |
| GIC-DLC           |       **2.74** |
| MLP learned codec |           2.34 |

Thus, GIC-DLC outperforms the tested traditional codecs on the training distribution, but the MLP version still compresses better.

This shows the main trade-off:

```text
MLP:
better compression
but expensive computation

DLC:
slightly worse compression
but much cheaper inference
```

On out-of-distribution datasets, performance deteriorates:

* KMNIST: 4.16 bpp,
* Fashion-MNIST: 6.27 bpp.

JPEG-XL performs better on both.

Therefore, the learned codec is strongly specialized to its training distribution. 

# 12. Importance of the learned upsampling model

The authors compare learned UPS with bicubic interpolation.

The learned logic-based upsampler produces lower reconstruction error.

This matters because a more accurate upsampling estimate means:

* smaller residual uncertainty,
* sharper ARM probability distributions,
* fewer encoded bits.

Figure 2 on page 4 shows that the finest level contains most of the bit cost and that learned upsampling reduces RMSE relative to bicubic interpolation. 

# 13. Energy and latency claims

The authors estimate approximately:

$
4\ \text{nJ/pixel}
$

for both encoding and decoding, compared with:

* PNG encoding: (322.58) nJ/pixel,
* PNG decoding: (39.19) nJ/pixel.

They therefore claim roughly:

* (100\times) lower encoding energy,
* (10\times) lower decoding energy.

They also estimate less than:

$
5\ \text{ns/pixel}
$

for encoding and decoding. 

However, these are **not direct hardware measurements**.

The authors explicitly state that:

* they did not implement GIC-DLC on an FPGA,
* energy is extrapolated from related LUT-network hardware,
* latency is estimated from other NeuraLUT results.

Therefore, these results should be interpreted as projections, not demonstrated hardware performance.

# 14. Relation to DLGN, IWP, and WNNs

## Compared with original DLGN

Original DLGN:

* 2-input gates,
* 16 candidate Boolean functions,
* softmax gate selection,
* typically classification.

GIC-DLC:

* 6-input LUTs,
* 64-entry truth tables,
* small neural networks train the LUT contents,
* learned connectivity,
* regression of distribution parameters for compression.

## Compared with IWP

IWP directly learns truth-table entries for a small gate.

GIC-DLC also ends with truth tables, but uses a neural network as the soft LUT representation during training.

Thus:

```text
IWP:
directly parameterize truth-table values

NeuraLUT:
use an internal neural network
to generate the truth-table function
```

IWP is more direct, while NeuraLUT can potentially represent structured multi-input LUT behavior more scalably.

## Compared with weightless neural networks

This paper is very close to WNN concepts because:

* neurons are LUTs,
* inputs form a LUT address,
* inference is memory lookup,
* connectivity is learned.

It is arguably closer to a differentiable WNN than to a standard DLGN.

# 15. Critical assessment

The interesting contribution is demonstrating that logic/LUT networks can be used not only for classification, but as **probability models inside a learned codec**.

That is meaningful because image compression requires continuous-value prediction:

$
\mu,\sigma,
$

rather than just Boolean or class outputs.

The strongest ideas are:

* hierarchical multi-resolution design,
* logic-based learned upsampling,
* autoregressive probability prediction,
* learned 6-input LUTs and connections,
* end-to-end optimization for bitrate.

The main limitations are substantial:

1. **Toy datasets only.** EMNIST, KMNIST, and Fashion-MNIST are (28\times28) grayscale images, not realistic natural-image compression workloads.

2. **Poor out-of-distribution behavior.** Compression performance drops sharply outside the EMNIST distribution.

3. **No real FPGA implementation.** Hardware benefits are estimated rather than measured.

4. **MLP remains more accurate as a probability model.** The MLP codec achieves lower bits per pixel.

5. **Thermometer encoding is expensive.** One grayscale value becomes 255 binary indicators.

6. **Autoregressive decoding remains sequential.** Even with fast LUT operations, pixel dependencies may constrain full-image throughput.

7. **The entropy coder is still required.** The entire system is not purely Boolean LUT inference; ANS introduces additional integer arithmetic and state.

# Bottom line

This paper can be summarized as:

$
\boxed{
\text{NeuraLUT logic networks}
+
\text{learned multi-resolution image prediction}
+
\text{autoregressive entropy modeling}
+
\text{lossless ANS coding}
}
$

The paper is important because it expands differentiable logic circuits from classification into **learned data compression**. However, it is currently a proof of concept on simple grayscale datasets, with projected rather than measured hardware efficiency.
