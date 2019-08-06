# The rise of robot authors

Movie production industry is going to face a skills shortage of unprecedented levels by 2030. Demand for screenwriters greatly outweighs the supply. A major production company found that the best solution is to invest in AI solution to generate movie scripts for all its new movies. Based on this fictitious storyline, we will build a Recurrent Neural Networks (RNN) algorithm using pytorch to automatically generate movie script based on specific genre.  

## Project overview
Our goal in this project is to generate movie script using a Recurrent Neural Network. We will train the network on existing scripts of specific genre. Then we will use it to generate an original piece of writing. We will start by preprocessing and batching our data. Then we will use a constructor and other methods (forward and backprob) to build our model. Next, we will train our model and define hyperparameters. Finally, we will use our model to generate movie scripts from scratch.

The main challenge in this project is to identify the most suitable hyperparameters to create highly optimized text generator. Later we will experiment with different hyperparameters in order to optimize model training and minimize loss.

![Robot screenwriters](/images/RNN1.jpg)

### Steps

* [x] **Preprocessing Data**: we should implement `create_lookup_tables` function and create a special token dictionary `token_lookup`
* [x] **Batching Input Data**:
  * Use `batch_data` to break data into sequences
  * Create data using TensorDataset
  * Batch data correctly.
* [x] **Build the RNN classes**: we will build `__init__`, `forward` , and `init_hidden` functions.
* [x] **RNN Training**:
  * Hyperparameters: we need to select suitable hyperparameters for training (below)
  * Improve loss: our goal is to reduce loss at least below 3.5.
* [x] **Generate movie Script**: generated script should look similar to the script genre in the dataset.

## Defining Hyperparameters

### Experiment structure:
Selecting optimal parameters for neural network architecture can often make the difference between mediocre and state-of-the-art performance. There are no magic hyper parameters that work everywhere. The best numbers depend on each task and each dataset. We will explore different hyperparameters and their impact on model training and we will experiment with these parameters until we are able to identify best figures that produce most optimized results.

* My experiment will have the following structure (rows):
  * Benchmark (initial parameters): I started with random parameters or parameters from similar published numbers and used them as my benchmark.
  * Experiment(s): then I start experimenting and observing impact of each parameter on convergence, time and GPU utilization to identify most suitable parameters.
  * Conclusion: findings and conclusions.

* Columns:
  * Hyperparameter column: the first column represents the hyperparameter that we are using in our experiment.
  * Result columns: the remaining column with contain results (Loss, Time and GPU utilization) from using that hyperparameter in training.

* Note- epochs:
  * Experimentation: during experimentation phase, I only used 2 epochs for each parameter.
  * Final model: Later, for the final model (with most suitable parameters defined), I’ve increase epoch to 20.

---
### Data parameters:

The most challenging part of building a network is defining proper batches. The following are my “experiment findings” for sequence length and batch size:
#### *Sequence length*:

| | Length | Loss | Time | GPU utilization |
| - | - | - | - | - |
| Benchmark | 10 | 3.933 |  5:37  | 60-70%  |
| Experiment1 |  40       | 3.917 | 13:22 | 80-90% |
| Conclusion   |||| limited improvement in loss but with huge increase in both time and GPU utilization. I decided to set sequence_length = 10

#### *Batch size*:

When selecting batch size, low size (32) will result in too slow model. Large size (256) although faster, but it’s computationally taxing and could result in worse accuracy.

| | Size | Loss | Time | GPU utilization |
| - | - | - | - | - |
| Benchmark | 256 | 3.933 | 5:37 | 60-70% |
| Experiment1 | 128 | 3.893 | 8:17 | 55-60% |
| Conclusion |||| this was my last experiment to perform, because I want to increase GPU utilization and reduce time while experimenting with other parameters. Nevertheless, I decided to use batch size=128 as my model parameter, because it converged faster than 256. |

---
### Training parameters:

learning_rate, embedding_dim, hidden_dim and n_layers
#### *Learning rate*:

LR is the most important hyper parameter. A good starting point = 0.01. However, other suspects are between: 0.1- 0.000001

| | lr | Loss | Time | GPU utilization |
| - | - | - | - | - |
| Benchmark | 0.001 | 3.933 | 5:37 | 60-70% |
| Experiment1 | 0.01 | 4.176 | 5:39 | 65-70% |
| Experiment2 | 0.0001 | 4.419 | 5:33 | 60-70% |
| Conclusion |||| lr= 0.001 has the best loss and I decided to keep it. |

#### *embedding_dim*:

It will define the depth of our embedding vector. The performance of some tasks improves the larger we make the embedding. However, any value between 200 and 500 would work.

| | embed_dim | Loss | Time | GPU utilization |
| - | - | - | - | - |
| Benchmark | 400 | 3.933 | 5:37 | 60-70% |
| Experiment1 | 200 | 3.970 | 5:33 | 60-65% |
| Experiment2 | 500 | 3.926 | 5:41 | 65-70% |
| Conclusion |||| the larger embed, it improves loss, but with little increase in both training time and GPU utilization. I decided to keep embed_dim= 400. |

#### *hidden_dim*:

It represents the number of units in the hidden layers of our LSTM cells. The main requirement is to set a number of hidden units that is “large enough”. Larger values basically allow a network to learn more text features but will increase overfitting. Common values are 128, 256, 512, etc.

| | hidden_dim | Loss | Time | GPU utilization |
| - | - | - | - | - |
| Benchmark | 256 | 3.933 | 5:37 | 60-70% |
| Experiment1 | 512 | 3.799 | 7:55 | 75-80% |
| Conclusion |||| increasing hidden layer features has improved our loss drastically but it was computationally (time + GPU) taxing. I decided to set hidden_dim= 512 because loss reduction from increasing hidden_dim was much better than loss reduction from increasing n_layer (below). Since increasing model capacity (both hidden_dim + n_layer) will increase overfitting, I decided to keep n_layer and only increase hidden_dim |

#### *n_layers*:
Represents the number of recurrent layers in our model

| | n_layers | Loss | Time | GPU utilization |
| - | - | - | - | - |
| Benchmark | 2 | 3.933 | 5:37 |60-70% |
| Experiment1 |3 | 4.099 | 6:44 | 60% |
| Conclusion |||| I didn’t find any improvement (in my case) from increasing n_layer. I decided to keep n_layer =2. |

---




### Results
Our model was able to generate the following script:
```
kramer: crowded liquid.

jerry: what?

jerry: i don't know.

george:(shouting) i have to tell you that. i can't even have to do it.

jerry: well, i think i'm getting a free sample.

elaine:(to jerry) i told you not to tell you this thing!

jerry: i don't know if i was eighteen- worthy.

elaine: oh, well, that's the thing.

george: what are you talking about?(he picks it up) i was in the sauna with my cousin holly.

kramer: oh, no!

george: hey.

elaine: oh.

jerry:(to kramer) so, i guess i could help you out. i'm gonna get going.

jerry:(to jerry) hey, i gotta tell you, i'm sorry to disturb you.

jerry:(to the intercom) hello? yeah, tommy fries is...

george:(interrupting) i don't wanna see this. i don't know what to do with it.

kramer: oh.

george:(to george, to himself) you know, you should be ashamed of yourself!

elaine: oh my god, you know, i really think you're gonna be the one who wants you to do that.

jerry: i think i could.

elaine:(smiling, convincing) oh, you idiots!

jerry:(to the phone) hello?(listens) yeah, yeah i know. i gotta tell you.(exits)

jerry: hey, i gotta skedaddle. i gotta get my mail cleaned.

jerry: oh, i don't want to talk to him.

kramer: hey.

jerry: hey.

kramer: hey, hey, you know...............

jerry: oh.

elaine: oh, come on, jerry, please. i don't want to sit down there
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Reference

Udacity deep learning program (Capstone project)
