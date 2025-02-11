# Bayesian Neural Network for Satellite Image Classification

This project implements a **Bayesian Neural Network (BNN)** using **SWAG (Stochastic Weight Averaging-Gaussian)** for satellite image classification. The approach provides well-calibrated uncertainty estimates and can identify ambiguous test samples.

## Key Features
### SWAG Implementation
- Tracks weight statistics during training to fit a Gaussian posterior:
  ```math
  \theta \sim \mathcal{N}(\mu_{\text{SWAG}}, \Sigma_{\text{SWAG}})
  ```
- Supports **SWAG-Diagonal** and **Full SWAG** methods.

### Calibration
- Evaluates calibration using the **Expected Calibration Error (ECE)**:
  ```math
  \text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
  ```
  where:
  - $`B_m`$ is the set of predictions in bin $`m`$,
  - $`\text{acc}(B_m)`$ is the empirical accuracy in bin $`m`$,
  - $`\text{conf}(B_m)`$ is the average confidence in bin $`m`$.

### Prediction Cost
- Implements an asymmetric cost function:
  ```math
  \ell(y, \hat{y}) = 
  \begin{cases} 
    1 & \text{if } \hat{y} = -1 \\
    3 & \text{if } \hat{y} \neq y \text{ and } \hat{y} \neq -1 \\
    0 & \text{if } \hat{y} = y
  \end{cases}
  ```

## Dataset
- **Training**: 1800 images (60x60 RGB) with well-defined labels from six land usage types.
- **Validation**: Includes well-defined and ambiguous samples for calibration.
- **Test**: Contains ambiguous or unseen combinations of land usage.

## Results
- Bayesian Model Averaging (BMA) provides robust predictions:
  ```math
  p(y=j | x) = \frac{1}{N} \sum_{i=1}^{N} p(y=j | x, \theta_i)
  ```
- Visualizations include **reliability diagrams** and confidence-based prediction samples.
- **Final overall cost**: 0.837 (ranked 233/275)
