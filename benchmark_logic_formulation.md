# Benchmark Logic Formulation

This document summarizes the mathematical formulation of the current benchmark logic.

## Data Setup

For each home \(h\), let the raw power series be \(p_t^{(h)}\) in watts and weather/features be \(z_t^{(h)}\).

The preprocessing used by the pipeline is:

\[
\tilde p_t^{(h)} = \log(1 + p_t^{(h)})
\]

Using only training homes, compute

\[
\mu_p,\sigma_p \quad\text{from all } \tilde p_t^{(h)}
\]

and similarly weather normalization stats

\[
\mu_w,\sigma_w
\]

Normalized power is

\[
x_t^{(h)} = \frac{\tilde p_t^{(h)} - \mu_p}{\sigma_p}
\]

For each sample, with history length \(L=43200\) and forecast horizon \(H=240\),

\[
X_{\text{past}} = (x_{t-L+1},\dots,x_t), \qquad
Y = (x_{t+1},\dots,x_{t+H})
\]

Because the model predicts 10-minute patches, targets are averaged over patch size \(P=10\):

\[
\bar y_j = \frac{1}{P}\sum_{i=1}^{P} y_{(j-1)P+i}, \qquad j=1,\dots,H/P
\]

So the model outputs quantiles

\[
\hat q_{j,\tau}, \qquad \tau \in \{0.1,0.5,0.9\}
\]

## Standard Pinball Loss

For one patch \(j\) and quantile \(\tau\),

\[
L_{\tau}(\bar y_j,\hat q_{j,\tau})
= \max\big(\tau(\bar y_j-\hat q_{j,\tau}),(\tau-1)(\bar y_j-\hat q_{j,\tau})\big)
\]

The standard benchmark loss is

\[
L_{\text{pinball}}
=
\frac{1}{NJT}\sum_{n,j,\tau} L_{\tau}(\bar y_{n,j},\hat q_{n,j,\tau})
\]

## Asymmetric Spike Quantile Loss

Let

\[
e_{j,\tau} = \bar y_j - \hat q_{j,\tau}
\]

Define extreme-event indicators in normalized space with threshold \(z_0=1.5\):

\[
I^{\text{peak}}_j = \mathbf 1[\bar y_j > z_0], \qquad
I^{\text{trough}}_j = \mathbf 1[\bar y_j < -z_0]
\]

Cowardly miss indicators:

- Peak miss = underpredicting a peak

\[
C^{\text{peak}}_{j,\tau} = I^{\text{peak}}_j \mathbf 1[e_{j,\tau} > 0]
\]

- Trough miss = overpredicting a trough

\[
C^{\text{trough}}_{j,\tau} = I^{\text{trough}}_j \mathbf 1[e_{j,\tau} < 0]
\]

Peak/trough magnitudes:

\[
M^{\text{peak}}_j = \max(\bar y_j - z_0, 0)
\]

\[
M^{\text{trough}}_j = \max(-z_0 - \bar y_j, 0)
\]

With separate weights \(W_{\text{peak}}, W_{\text{trough}}\), the penalty multiplier is

\[
\Gamma_{j,\tau}
=
\max\Big(
1 + C^{\text{peak}}_{j,\tau} W_{\text{peak}} M^{\text{peak}}_j,\;
1 + C^{\text{trough}}_{j,\tau} W_{\text{trough}} M^{\text{trough}}_j
\Big)
\]

Weighted pinball loss:

\[
\tilde L_{j,\tau} = \Gamma_{j,\tau} L_{\tau}(\bar y_j,\hat q_{j,\tau})
\]

Average across quantiles:

\[
\tilde L_j = \frac{1}{3}\sum_{\tau}\tilde L_{j,\tau}
\]

Then split timesteps into normal and extreme:

\[
\mathcal N = \{j: I^{\text{peak}}_j + I^{\text{trough}}_j = 0\}, \qquad
\mathcal E = \{j: I^{\text{peak}}_j + I^{\text{trough}}_j = 1\}
\]

Final asymmetric loss:

\[
L_{\text{asym}}
=
\frac{1}{|\mathcal N|}\sum_{j\in\mathcal N}\tilde L_j
+
\frac{1}{|\mathcal E|}\sum_{j\in\mathcal E}\tilde L_j
\]

with empty-set terms treated as \(0\).

## Benchmark Evaluation

Predictions are denormalized back to watts:

\[
\hat p_{j,\tau} = \exp\big(\hat q_{j,\tau}\sigma_p + \mu_p\big)-1
\]

\[
p_j = \exp\big(\bar y_j \sigma_p + \mu_p\big)-1
\]

Using the median forecast \(\hat p_{j,0.5}\):

MAE:

\[
\text{MAE} = \frac{1}{N}\sum_j |p_j - \hat p_{j,0.5}|
\]

wMAPE:

\[
\text{wMAPE} = 100 \cdot \frac{\sum_j |p_j - \hat p_{j,0.5}|}{\sum_j p_j + \varepsilon}
\]

Define peak and trough sets from the realized target distribution:

\[
\mathcal P = \{j : p_j \ge Q_{0.9}(p)\}, \qquad
\mathcal T = \{j : p_j \le Q_{0.1}(p)\}
\]

Peak absolute percentage error:

\[
\text{PAPE} =
100 \cdot \frac{1}{|\mathcal P|}\sum_{j\in\mathcal P}
\frac{|p_j - \hat p_{j,0.5}|}{p_j+\varepsilon}
\]

Trough absolute percentage error:

\[
\text{TAPE} =
100 \cdot \frac{1}{|\mathcal T|}\sum_{j\in\mathcal T}
\frac{|p_j - \hat p_{j,0.5}|}{p_j+\varepsilon}
\]

Peak coverage of the 90th quantile:

\[
\text{Cov}_{\text{peak}}^{90}
=
100 \cdot \frac{1}{|\mathcal P|}\sum_{j\in\mathcal P}\mathbf 1[p_j \le \hat p_{j,0.9}]
\]

Trough coverage of the 10th quantile:

\[
\text{Cov}_{\text{trough}}^{10}
=
100 \cdot \frac{1}{|\mathcal T|}\sum_{j\in\mathcal T}\mathbf 1[p_j \ge \hat p_{j,0.1}]
\]

## Peak/Trough Breakdown Plot

For peaks and troughs separately, with error

\[
\delta_j = \hat p_{j,0.5} - p_j
\]

and interval width

\[
w_j = \hat p_{j,0.9} - \hat p_{j,0.1}
\]

the plotted quantities are:

\[
\text{Peak MAE} = \frac{1}{|\mathcal P|}\sum_{j\in\mathcal P} |\delta_j|
\]

\[
\text{Peak Bias} = \frac{1}{|\mathcal P|}\sum_{j\in\mathcal P} \delta_j
\]

\[
\text{Peak Interval Width} = \frac{1}{|\mathcal P|}\sum_{j\in\mathcal P} w_j
\]

and analogously for \(\mathcal T\).

## Weight-Sweep Benchmark

The asymmetric-weight benchmark trains one model for each pair

\[
(W_{\text{peak}}, W_{\text{trough}})
\]

in the union of:

1. Trough-only sweep

\[
(0,0),(0,5),(0,10),(0,15),(0,20)
\]

2. Peak-only sweep

\[
(0,0),(5,0),(10,0),(15,0),(20,0)
\]

3. Diagonal sweep

\[
(0,0),(5,5),(10,10),(15,15),(20,20)
\]

After deduplication, the unique set is:

\[
\{(0,0),(0,5),(0,10),(0,15),(0,20),(5,0),(10,0),(15,0),(20,0),(5,5),(10,10),(15,15),(20,20)\}
\]

Each configuration is trained for 300 epochs, then evaluated with the metrics above.
