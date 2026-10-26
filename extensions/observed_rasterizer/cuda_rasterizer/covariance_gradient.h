#pragma once

// Forward values and public gradients remain float32. Only the optional
// covariance derivative reduction and its chain rule use wider accumulators.
#if defined(MY4DGS_STABLE_COVARIANCE_GRADIENTS)
using CovarianceGradientScalar = double;
#else
using CovarianceGradientScalar = float;
#endif
