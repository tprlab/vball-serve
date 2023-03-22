# Volleyball service detector with machine learning

## Goal

Using data about volleball movement (frame, x, y, size) in trdata.json, classify the trajectories. 

First priority is to recognize serves.

## Solution

Using lazypredict, compare different classifiers from sklearn.

Dockerfile is provided.

Also there is [an action](https://github.com/tprlab/vball-serve/actions/workflows/test_predict.yml) to run direcly on Github.

## Outcome 

Random forest is the winner!
