### From MNIST to ImageNet: Understanding the Scalability Boundaries of Differentiable Logic Gate Networks - Summary notes

In this paper, the authors do **not introduce a new gate parameterization such as IWP**. Instead, they study a different bottleneck:

> **Can the standard DLGN output layer scale from 10 classes to hundreds or thousands of classes?**

The paper focuses on the **Group-Sum output layer**, the temperature parameter (\tau), and large-class classification. 

## 1. What is the Group-Sum layer?

Suppose the final logic layer has (n) Boolean outputs and the task has (k) classes.

The authors divide the outputs into (k) groups:

```text
Outputs 1 ... n/k          → Class 1
Outputs n/k+1 ... 2n/k     → Class 2
...
Last group                 → Class k
```

For each class, they sum the active logic outputs:

$
s_c=\sum_{j\in G_c}o_j
$

and then apply a temperature-scaled softmax:

$
p_c=
\frac{\exp(s_c/\tau)}
{\sum_r\exp(s_r/\tau)}
$

The predicted class is the group with the largest sum. 

The concern is obvious: as the number of classes increases, each class gets fewer output neurons.

For example, with 64,000 outputs:

| Classes | Outputs per class |
| ------: | ----------------: |
|      10 |             6,400 |
|     100 |               640 |
|   1,000 |                64 |
|   2,000 |                32 |

The authors investigate whether this eventually prevents DLGNs from distinguishing many classes.

## 2. What do they test?

They evaluate DLGNs with up to **2,000 classes** using:

* a synthetic binary dataset,
* combined MNIST-family datasets with up to 67 classes,
* CIFAR-10 and CIFAR-100,
* ImageNet-32 with up to 1,000 classes.

They compare DLGNs against MLPs and also perform supplementary experiments with convolutional DLGNs. 

## 3. Their main finding: temperature matters enormously

The most important result is that the Group-Sum layer is much more sensitive to (\tau) than previous papers implied.

The temperature does not change the final argmax directly, because dividing every class score by the same positive number preserves their ordering. However, during training it changes:

* softmax confidence,
* gradient magnitude,
* how information is distributed across output neurons,
* whether a few neurons dominate each class.

### Small (\tau)

A small temperature makes the class softmax sharp:

$
\operatorname{softmax}(s/\tau),\qquad \tau\ll1
$

Small score differences become large probability differences.

This can create:

* unstable training,
* overconfidence,
* class dependence on a few output neurons,
* many dead or saturated neurons.

### Large (\tau)

A larger temperature makes the class probabilities smoother.

This tends to:

* distribute responsibility across more neurons,
* reduce domination by individual outputs,
* create more ensemble-like behavior,
* make pruning less damaging.

But if (\tau) is too large, predictions and gradients become too weak.

The optimal value therefore depends strongly on the task and particularly on the **number of output neurons per class**. 

## 4. Why does the number of classes affect the best temperature?

Consider two cases.

### Ten classes

With 64,000 outputs, each class receives 6,400 neurons. The sums can become very large, and relatively small percentage differences can produce huge absolute score differences.

A larger (\tau) is useful because it moderates these sums.

### Two thousand classes

Each class receives only 32 neurons. Class sums are much smaller and closer together.

A lower (\tau) is then useful because it amplifies the distinction between these small sums.

Their empirical pattern is approximately:

```text
Many neurons per class → larger τ
Few neurons per class  → smaller τ
```

This is one of the paper's most concrete design lessons.

## 5. Does Group-Sum actually scale?

On the simple synthetic dataset, yes.

The authors show that DLGNs can classify up to 2,000 classes and can outperform the tested MLP when (\tau) and network capacity are chosen appropriately. Importantly, even **32 output neurons per class** can be enough on that structured dataset. 

On combined MNIST datasets, DLGNs and convolutional DLGNs remain competitive up to 67 classes when (\tau) is tuned appropriately.

However, on ImageNet-32, the result is much weaker. Accuracy drops sharply as the class count grows, and the DLGN does not match the MLP beyond relatively small numbers of classes. 

## 6. Their diagnosis of the ImageNet failure

The authors argue that the output layer is **not necessarily the main limitation**.

A six-layer binary DLGN neuron can depend on at most approximately:

$
2^6=64
$

original signals, assuming no duplicated paths.

For ImageNet-32, the threshold-expanded input has 9,216 dimensions. Thus, one output neuron sees information originating from only about:

$
\frac{64}{9216}\approx0.7%
$

of the input.

That is probably insufficient for complex natural images. Increasing the output layer size does not fix this, because the feature-extraction backbone still has a very limited receptive field. 

This is an important conclusion:

> DLGN scalability is limited not only by class decoding, but also by sparse random connectivity, shallow dependency depth, and restricted input coverage.

## 7. Output-layer alternatives

The authors also test alternatives to Group-Sum, including:

* binary losses at the individual output level,
* a fully connected final layer,
* fitting a fully connected layer after DLGN training,
* codebook-based class prediction using Hamming distance,
* combinations of Group-Sum and codebooks.

The codebook idea assigns every class a binary target code. The network outputs a binary vector, and classification selects the nearest class code by Hamming distance.

This can improve some datasets and reduces sensitivity to (\tau), but no alternative consistently dominates Group-Sum across all experiments. 

## 8. Their analysis of output-neuron redundancy

They inspect the activation rate of each output neuron: the fraction of samples for which the neuron outputs 1.

With low (\tau), many neurons cluster around activation rates of:

* 0%: effectively dead,
* 100%: always active,
* approximately 50%: potentially synchronized or redundant.

With higher (\tau), activation rates are more broadly distributed. The authors interpret this as better differentiation among neurons and more distributed class information. They then show that these networks tolerate random output-neuron pruning better. 

This is related to, but different from, the “unused neurons” discussion in *Mind the Gap*:

* *Mind the Gap* studies uncertainty over **gate choices**.
* This paper studies redundancy in **output activations and class evidence**.

## 9. Relation to IWP

This paper appears to use the original 16-way gate mixture rather than IWP.

Therefore:

* IWP improves the internal representation and optimization of each logic neuron.
* This paper studies how the final neurons are decoded into class predictions.

They address different levels:

```text
IWP:
How should each gate be parameterized?

This paper:
How should thousands of Boolean outputs represent many classes?
```

A future model could use IWP internally while adopting the output-layer lessons from this paper.

## 10. Critical assessment

The strongest contribution is the systematic demonstration that **(\tau) is a fundamental architectural hyperparameter**, not a minor softmax setting. The authors also show that Group-Sum itself is more scalable than previously assumed on structured data.

However, the title is broader than the demonstrated advance. They do not solve ImageNet-scale DLGNs. Their real-world results reveal that the major bottleneck may be the backbone's limited receptive field and input coverage, not merely the output layer.

The paper's central conclusion is therefore:

> Group-Sum can represent hundreds or thousands of classes when features are sufficiently separable and (\tau) is tuned, but current DLGN backbones still struggle to extract globally informative features from complex natural images.
