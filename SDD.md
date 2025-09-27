# Neural Belief Propagation Ensemble Decoder (NBED) - Software Design Document

## Overview

The Neural Belief Propagation Ensemble Decoder (NBED) is combines multiple NBP4 decoders into an ensemble for improved performance on toric and Generalized Bicycle (GB) codes.

## Architecture

### Core Components

1. **NBED.py** - Main neural belief propagation decoder implementation, consolidated from multiple components in the repository
2. **simulateFER.cpp** - C++ framework for evaluating ensemble as well as single decoders

## Training New Decoders

To train a new neural decoder, add the following code in **NBED.py**:

```python
decoder = init_and_train(
    n=254, k=28, m=254, 
    n_iterations=6, 
    error_weights=(2,3),
    codeType='GB',
    params=training_configs['paper'],
    # This is the name that will be used by the C++ code
    name="name"
)
```
To prune the decoder simply call
```python
decoder.prune_weights(0.3)
train(decoder)
```
### Evaluating Ensemble

Add your decoder to the ensemble in **simulateFER.cpp**

```cpp
fileReader first_matrix_supplier(n, k, m, codeType, trained, "first-decoder");
matrix_supplier.check_symplectic();
```
and inside the multi-threaded decoding loop

```cpp
std::vector<bool> success;
// Create the same error pattern for all paths, the supplied matrix does not
// have to be from a path in the ensemble
stabilizerCodes errorCreator(n, k, m, codeType, first_matrix_supplier, trained);
errorCreator.add_error_given_epsilon(epsilon);

ensembleDecoder ens;
stabilizerCodes first(n, k, m, codeType, first_matrix_supplier, trained);

ens.add_decoder(first);
ens.setErrors(errorCreator.getErrorString(), errorCreator.getError());
```
Finally, set the desired metric, either NBED:

```cpp
// NBED Curve
success = ens.decodeAllPaths(decIterNum, ep0);
```
or List-error-rate (LER) NBED 

```cpp
// LER-NBED curve
for(int i = 0 ; i < ens.list_of_decoders.size(); i++){
    success = ens.list_of_decoders[i]->decode(decIterNum, ep0);
    if (success[1]) {
        break;
    }
}
```


