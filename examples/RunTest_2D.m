clear, clc,close all

pyenv(ExecutionMode="OutOfProcess");
addpath("./python_func")
addpath("./MMA")

% Specify geometry and optimization parameters
nelx = 250;
nely = 70;
holes = 0;
vol = 0.3;
penal = 2.5;
rmin = 3;
ft = 2;
i = 0.7;


top110NN(nelx,nely,vol,penal,rmin,ft,holes,500,i,1);

