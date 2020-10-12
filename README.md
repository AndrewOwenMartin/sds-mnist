# Demo

See a demo at [Asciinema](https://asciinema.org/a/326628 "Asciinema").

# Introduction

In the demo you will see a user clone the repository, set up the projects, gather the MNIST database of handwritten digits, and run three examples. For each example an image is selected at random from the test dataset, the image is rendered to the terminal and the algorithm returns the integer it thinks is represented by the image.

Note there is no training phase.

The underlying algorithm (Stochastic Diffusion Search - SDS)  was described in our discussion.

 - **DIFFUSION PHASE** - Any inactive agent polls an agent at random if the polled agent is active the inactive agent copies the hypothesis of the polled agent. If the polled agent was inactive the inactive agent selects a hypothesis at random.
 - **TEST PHASE** - Each agent selects a microtest at random and applies it to their hypothesis. If the test passes the agent becomes active, if the test fails the agent becomes inactive.

In this case a hypothesis is an image in the training set.

A microtest is as follow. Select two pixels A and B at random from the test image, set a boolean value X to True if A is 'brighter' than B, else false. Select the two equivalent pixels from the agent's hypothesis image, set a boolean value Y to True if A is 'brighter' than B. The test passes if X equals Y.

Within a few iterations of applying the **TEST PHASE** and **DIFFUSION PHASE** to all agents, there will be a 'cluster' of agents maintaining the same hypothesis, this will be the image from the training set which is most similar to the image from the test set. We return the number of the image from the training set.

# Dependencies

Expects wget and gunzip to be on your path, and expects numpy and sds to be in your python environment.

```
pip install numpy sds
./get_datasets.sh
```

then run either

```
python image_similarity.py example
```

or

```
python image_similarity.py experiment
```
