### Two-Stage Unit Tying for Simplifying Differentiable Logic Gate Networks - Summary notes

In this paper, the authors introduce a **post-training simplification method for DLGNs** called **two-stage unit tying**.

Their goal is very practical:

> A trained DLGN may be too large to fit on a target FPGA. Can we simplify the already-trained logic circuit while preserving as much accuracy as possible?

The method is analogous to neural-network pruning, but instead of deleting weights, the authors force selected logic units to output a constant:

$
0 \quad\text{or}\quad 1.
$

This enables hardware synthesis tools to propagate constants through the network and eliminate downstream logic. 

# 1. What is “unit tying”?

Consider a learned gate:

$
y = A \oplus B.
$

The authors may replace it with:

$
y=0
$

or:

$
y=1.
$

This appears wasteful because the unit is no longer performing useful computation. However, the hardware simplification can propagate much farther.

Suppose:

$
z=y\land C.
$

If (y) is tied to zero:

$
z=0\land C=0.
$

Now the synthesis tool can eliminate:

* the original XOR;
* the AND;
* possibly later gates receiving (z).

Thus, tying one unit can trigger a cascade of circuit removal:

```text
Selected gate → constant
        ↓
constant propagation
        ↓
downstream simplification
        ↓
fewer FPGA LUTs
```

The diagram on page 1 illustrates this clearly: the original model exceeds the FPGA budget, while progressively tying units reduces LUT utilization until the design fits. 

# 2. Is this the same as pruning?

Conceptually, yes, but mechanically, no.

## Conventional pruning

A weight is changed from a small real value to zero:

$
w=0.003\rightarrow0.
$

This is often a relatively small perturbation.

## Unit tying

A gate such as XOR or AND is replaced by a constant:

$
\operatorname{XOR}(A,B)\rightarrow0.
$

This is a major functional change.

The authors show on page 3 that approximately 60% of candidate units require an almost complete reassignment of their gate probability to become a constant. Therefore, standard Taylor-based pruning approximations may be unreliable because tying is not a small local perturbation. 

This observation motivates their two-stage method.

# 3. What objective do they preserve?

They do not initially measure the classification loss against the labels.

Instead, they treat the original unmodified model as a **teacher**.

For an input (x), let:

$
z_{\emptyset}(x)
$

be the original network logits and:

$
z_{S,t}(x)
$

be the logits after tying the units in set (S).

They measure:

$
\mathcal L_{\mathrm{dist}}(S,t)
===============================

\mathbb E_x
\left[
\frac12
\left|
z_{S,t}(x)-z_{\emptyset}(x)
\right|_2^2
\right].
$

In plain language:

> Select units whose removal changes the original network’s output scores as little as possible.

This is a function-preservation objective resembling knowledge distillation, but it is used to decide which gates to simplify—not merely for later retraining. 

# 4. Why is selecting the units difficult?

Suppose a layer contains:

$
C=10{,}000
$

candidate units and the authors want to tie:

$
T=3{,}000.
$

They must decide:

1. which 3,000 units to tie;
2. whether each should become 0 or 1.

The number of combinations is enormous:

$
\binom{10{,}000}{3{,}000}2^{3{,}000}.
$

They cannot test all possibilities.

Even testing every unit individually with a full forward pass is expensive for wide networks. Therefore, they propose two stages:

```text
Stage A:
fast approximate screening

Stage B:
more accurate refinement
```

# 5. Stage A: Gauss–Newton screening

Each original DLGN unit has a distribution over the 16 Boolean functions:

$
p_u\in\mathbb R^{16}.
$

To tie the unit to constant (c\in{0,1}), this distribution is changed toward a one-hot vector:

$
e_c.
$

The change is:

$
\delta p_{u,c}=e_c-p_u.
$

The authors approximate how this modification changes the output logits using a Gauss–Newton quadratic score:

$
\hat s(u,c)
===========

\frac12
\mathbb E_x
\left[
\left|
J_{x,u}\delta p_{u,c}
\right|_2^2
\right],
$

where (J_{x,u}) measures how sensitive the output logits are to the unit’s gate distribution.

For each unit, they calculate:

$
s(u)=\min_{c\in{0,1}}\hat s(u,c)
$

and select whether it is safer to tie it to 0 or 1:

$
t(u)=\arg\min_{c\in{0,1}}\hat s(u,c).
$

Units with the smallest scores are expected to have the least effect on the network.

## Why Gauss–Newton?

The distortion objective is a squared logit error. This allows the authors to use Jacobian products rather than explicitly computing a full Hessian.

One backward pass gives information about many units simultaneously, making Stage A much cheaper than individually evaluating every possible tie. 

# 6. Why do they deliberately select too many units?

Suppose the final target is:

$
T=1{,}000
$

tied units.

Stage A does not select exactly 1,000. It selects:

$
T+k
$

units, where (k) is an overshoot margin—40 in their default experiments.

Thus, Stage A produces:

$
1{,}040
$

candidates.

Why overshoot?

Because the Gauss–Newton score is only an approximation. Some harmful units may incorrectly appear safe.

The extra candidates give Stage B room to identify and remove these mistakes.

This is shown in Figure 3 on page 4:

```text
All units
   ↓
Gauss–Newton screening
   ↓
overshoot set T+k
   ↓
refinement removes k harmful candidates
   ↓
final set T
```



# 7. Stage B: Binary-split refinement

This is the most distinctive part of the method.

The authors divide the current candidate set into two halves:

$
S=S_1\cup S_2.
$

They then perform two actual finite-difference evaluations:

### Evaluation 1

Tie (S_1), but leave (S_2) untied.

Measure distortion:

$
\mathrm{err}_1.
$

### Evaluation 2

Tie (S_2), but leave (S_1) untied.

Measure:

$
\mathrm{err}_2.
$

If:

$
\mathrm{err}_1>\mathrm{err}_2,
$

then the more harmful unit is assumed to lie inside (S_1).

They recursively split (S_1) and repeat until one harmful unit is isolated.

That unit is removed from the candidate set.

The procedure is repeated (k) times until only (T) units remain.

This resembles binary search or group testing:

```text
Candidate set
  ├─ first half
  └─ second half
       ↓
test both halves
       ↓
follow more harmful half
       ↓
repeat
       ↓
isolate one harmful unit
```

The method requires approximately:

$
O\left(k\log(T+k)\right)
$

finite-difference evaluations rather than evaluating every candidate individually. 

# 8. Why can Stage B be better than evaluating each unit separately?

Because tying effects are not necessarily additive.

Suppose tying unit (A) alone is harmless and tying unit (B) alone is harmless, but tying both breaks an important path.

A per-unit score may estimate:

$
\text{cost}(A)+\text{cost}(B)
$

and miss the interaction.

Stage B evaluates groups of units tied simultaneously, so it can capture some interaction effects.

The authors show that their Binary Split refinement performs better than:

* rescoring only the candidates near the ranking boundary;
* individually rescoring the entire overshoot set.

At 30% tying on CIFAR-10, Binary Split achieves 67.30% accuracy, compared with 66.33% for tail rescoring and 66.88% for full-set rescoring, while remaining much faster than full-set rescoring. 

# 9. How do they apply tying in convolutional DLGNs?

The paper uses the convolutional LogicTreeNet architecture.

For ordinary fully connected-like logic layers, each individual gate can be a tying candidate.

For convolutional logic-tree layers, they use **structured tying**:

> They tie only the root gate of each logic tree.

Tying the root effectively makes the corresponding output channel constant.

Thus, this is analogous to channel pruning in CNNs:

```text
CNN channel pruning:
remove one feature-map channel

Logic-tree tying:
force one tree root/output channel to 0 or 1
```

This avoids irregularly deleting internal gates while retaining the channel structure.

However, it may not maximize circuit reduction because internal subtrees that could be individually simplified are not directly considered.

# 10. Main accuracy results

For the medium CIFAR-10 model, the original accuracy is:

$
71.57%.
$

Without fine-tuning:

| Tied ratio | Their accuracy |
| ---------: | -------------: |
|        10% |         70.76% |
|        20% |         69.52% |
|        30% |         67.30% |
|        40% |         65.05% |
|        50% |         61.44% |

At 50% tying, simpler baselines perform much worse:

* constant-probability score: 24.73%;
* Fisher-based tying: 53.97%;
* finite-difference per-unit baseline: 56.05%;
* proposed method: 61.44%.

After fine-tuning, their 50%-tied CIFAR-10 model recovers to:

$
69.22%,
$

only 2.35 percentage points below the original model. 

For MNIST, the results are much stronger.

The original accuracy is:

$
98.91%.
$

At 50% tying:

* before fine-tuning: 96.20%;
* after fine-tuning: 98.74%.

So the network can tie half its selected units with only:

$
0.17
$

percentage-point loss after fine-tuning.

# 11. Does tying 50% of units reduce LUT use by 50%?

No.

This is an important distinction.

For CIFAR-10:

| Tied ratio | LUT reduction |
| ---------: | ------------: |
|        10% |         6.77% |
|        20% |        17.39% |
|        30% |        27.44% |
|        40% |        37.82% |
|        50% |        48.33% |

For MNIST, 50% tying gives:

$
42.88%
$

LUT reduction.

The relationship is not exactly linear because synthesis performs:

* constant propagation;
* Boolean simplification;
* common logic optimization;
* removal of unused downstream paths.

Some tied units produce little additional simplification, while others trigger larger cascades.

This confirms why gate count alone is not enough—the actual post-synthesis LUT count matters. 

# 12. The strongest practical result: smaller FPGA deployment

The original CIFAR-10 model uses approximately:

$
475{,}000
$

FPGA LUTs.

It fits only on the large Virtex-7 V7-2000T, listed at approximately $22,000 in the paper.

It does not fit on:

* V7-585T;
* K7-410T.

After 30% tying, the design uses about:

$
344{,}600
$

LUTs and fits the medium V7-585T.

After 50% tying, it uses:

$
245{,}400
$

LUTs and fits the smaller K7-410T at approximately 97% utilization.

Thus, the paper demonstrates a real qualitative change:

```text
Original model:
cannot be deployed

After tying:
fits the target FPGA
```

The reported latency remains 9 ns per image because the same pipelined schedule and timing constraint are retained. 

# 13. Is the method synthesis-aware?

Only partially.

The authors call unit tying synthesis-aware because the simplification mechanism relies on constant propagation and they evaluate actual synthesized FPGA LUT counts.

However, the selection objective does **not directly include synthesis cost**.

They optimize:

$
\text{logit preservation under a fixed tying ratio},
$

not:

$
\frac{\text{accuracy loss}}{\text{actual LUT reduction}}.
$

Every selected unit receives equal treatment in the tying budget, although different units may produce very different hardware savings.

For example:

* tying one root may eliminate a large downstream tree;
* tying another may save almost nothing because the synthesis tool already simplifies it;
* some units may have much greater fanout.

Their method attempts to find the least harmful (T) units, but not necessarily the units providing the best hardware saving per unit of distortion.

Therefore, a more accurate description is:

> **Function-preserving unit tying followed by synthesis-based evaluation**, rather than fully synthesis-cost-aware selection.

# 14. Relation to *Mind the Gap*

This is not the same as the “unused neuron” reduction discussed in *Mind the Gap*.

## Mind the Gap

The authors improve training and discretization so that the network does not waste neurons through uncertain or ineffective gate choices.

## This paper

The authors begin with an already trained and discretized network, then intentionally convert selected useful or less-useful units into constants to simplify the circuit.

So:

```text
Mind the Gap:
avoid ineffective neurons during training

Unit tying:
remove/simplify neurons after training
```

They address related inefficiency problems at different stages.

# 15. Relation to IWP, WARP, and CovJac

The paper uses the original 16-gate Soft-Mix formulation because its Stage-A perturbation is expressed in the 16-dimensional gate-probability space:

$
\delta p=e_c-p.
$

This makes tying to the constant-0 or constant-1 gates straightforward.

## IWP

IWP has four truth-table parameters. Tying to zero would mean:

$
[w_{00},w_{01},w_{10},w_{11}]
\rightarrow
[0,0,0,0].
$

Tying to one means:

$
[1,1,1,1].
$

The same concept is possible, but its Gauss–Newton screening would need to operate in the four-dimensional truth-table space.

## WARP

A constant function corresponds to a particular Walsh coefficient vector, so tying remains possible.

## CovJac

The codebook already contains the constant-0 and constant-1 gates. A natural extension would measure the distortion of moving the learned four-dimensional point toward either constant prototype.

Thus, unit tying is largely orthogonal to the gate parameterization. The method could be adapted to newer DLGN formulations.

# 16. Critical assessment

This paper addresses a genuine deployment problem that many DLGN papers ignore:

> A design that is slightly too large for an FPGA is not merely slower—it is undeployable.

Its strongest aspects are:

* a clear post-training complexity control mechanism;
* actual post-synthesis FPGA evaluation;
* an effective teacher-logit preservation objective;
* a computationally efficient two-stage selection method;
* demonstration that models can move from out-of-capacity to feasible;
* strong MNIST compression with negligible accuracy loss.

The main limitations are important.

### 1. It selects units by tied ratio, not actual hardware benefit

All units count equally toward (T), although their LUT-saving effects differ.

A stronger approach would optimize:

$
\frac{\Delta\text{LUTs}}{\Delta\text{distortion}}
$

or directly target a required FPGA LUT budget.

### 2. The refinement is heuristic

Binary Split assumes that repeatedly following the more harmful half will isolate a harmful unit. With strong multi-unit interactions, the globally harmful subset may not be recoverable through this procedure.

There is no guarantee that the final set is optimal.

### 3. Fine-tuning is expensive

Their post-tying recovery uses 30,000 fine-tuning iterations. The method is therefore not simply a calibration-only, no-training compression procedure.

The strong 50%-tied results depend heavily on this fine-tuning.

### 4. The first convolution and final classifier are excluded in accuracy experiments

They do not initially tie every layer. This protects sensitive components but leaves potentially important hardware overhead untouched.

For hardware experiments, they state that they apply tying to all layers, but the methodology and candidate structure for the first/final layers deserve clearer analysis.

### 5. Only FPGA LUT count is optimized and measured

They do not provide:

* dynamic power;
* energy;
* routing congestion;
* wirelength;
* flip-flop count;
* block RAM;
* critical-path variation beyond meeting the fixed timing constraint.

A smaller LUT count does not automatically imply proportionally lower power or easier routing.

### 6. The method remains tied to original Soft-Mix networks

The experiments do not test whether newer parameterizations—such as IWP, WARP, or CovJac—have different redundancy and tying behavior.

### 7. The models remain relatively low accuracy

The CIFAR-10 baseline achieves 71.57%. The method proves deployment feasibility for this DLGN, but not for a model competitive with modern CNNs.

# Bottom line

The paper introduces:

$
\boxed{
\text{Post-training DLGN simplification}
+
\text{constant unit tying}
+
\text{Gauss–Newton screening}
+
\text{binary-split finite-difference refinement}
}
$

The core idea is:

> Do not remove gates directly. Force selected gates to constant 0 or 1 so that synthesis can propagate those constants and eliminate entire downstream logic structures.

This is an important paper from a hardware-deployment perspective. It provides the DLGN equivalent of structured pruning and shows that post-training simplification can make an otherwise undeployable network fit on a much smaller FPGA.

The clearest research extension is to make the tying process **truly hardware-budget-aware**:

$
\boxed{
\text{select ties based jointly on}
\quad
\text{accuracy distortion}
+
\text{actual synthesis savings}
}
$

rather than selecting a fixed percentage of units and observing the LUT reduction afterward.
