# Perceptron Learning Algorithm

## Definition 
![Perceptron Learning Algorithm](img/perceptron-learning-alg.PNG?raw=true "Title")

## Example
Train a discrete bipolar perceptron using the Perceptron Learning Rule, taking:

    η = 1
    w₁ = [ 0  1  0 ]ᵗ (initial weights)
    x₁ = [ 2  1 -1 ]ᵗ, t₁ = -1
    x₂ = [ 0 -1 -1 ]ᵗ, t₂ =  1

Repeat the (x₁, t₁), (x₂, t₂) sequence until you obtain the correct outputs. We are using the signal function sgn for activation instead of g. The signal function is the same as g except that we have possible values of 1 and -1.

    First Epoch: w₁ = [ 0  1  0 ]ᵗ
    net₁ = w₁ᵗx₁ = [ 0  1  0 ] ⋅ [ 2  1 -1 ] = 1
    sgn(1) = 1

    w₂ = w₁ - 1 ⋅ (1 - -1) ⋅ x₁
    w₂ = [ 0  1  0] - 2 ⋅ [ 2  1 -1 ] = [ -4 -2  2 ]

    net₂ = w₂ᵗx₂ = [ -4 -2  2 ] ⋅ [ 0 -1 -1 ] = 0
    sgn(0) = -1

    w₃ = w₂ - 1 ⋅ (-1 - 1) ⋅ x₂
    w₃ = [ -4 -2  2 ] + 2 ⋅ [ 0 -1 -1 ] = [ -4 -4  0 ]

    Second Epoch: w₁ = [ -4 -4  0 ]ᵗ
    net₁ = w₁ᵗx₁ = [ -4 -4  0 ] ⋅ [ 2  1 -1 ] = -12
    sgn(-12) = -1

    Correction not performed since t₁ = -1.

    net₂ = w₂ᵗx₂ = [ -4 -4  0 ] ⋅ [ 0 -1 -1 ] = 4
    sgn(4) = 1

    Correction not performed since t₂ = 1.

Correct output obtained after 1 full epoch.