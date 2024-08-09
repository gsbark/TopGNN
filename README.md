# TopGNN

A topology optimization framework that leverages a Graph Neural Network (GNN) filter to map intermediate densities to solutions that are close to optimal to accelerate TO. 

## Dependencies

The following libraries are required:
| Package               | Version (>=) |
|-----------------------|--------------|
| numpy                 | 1.25.2       |
| torch                 | 2.0.1        |
| torch-geometric       | 2.3.1       |

## Usage

### 1. Set Up the Environment

Create a virtual enviroment containing all the dependecies and then configure Python interpreter in MATLAB.
```matlab
pyenv('Version', 'path_to_topgnn-env/python');

2. Running Examples and Training
1. Running a Benchmark 2D TO

To run a 2D topology optimization problem example, navigate to the `examples` folder and execute `RunTest_2D.m`

2. Training the Model:

To train the GNN model execute  `main_train.py` in the `GNN` folder
