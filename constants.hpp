#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <iostream>
#include <cmath>

const int NB_LAYERS = 3; //Number of layers of nodes, including the input and output layer

const int picture_size = 28 * 28;
const int NB_NODES_PER_LAYER[NB_LAYERS] = 
{picture_size, 30, 10};

const int training_set_size = 1000;
const int test_set_size = 5000;

const int nbEpochs = 30;
const int batchSize = std::min(training_set_size, 10);


const double oo = 1e8;

const double steepness = 1.0;
const double alpha = 5 * pow(10,-2);  //learning rate
#endif
