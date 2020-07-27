#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>
#include <cmath>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "utils.hpp"
#include "constants.hpp"

using namespace boost::numeric::ublas;


int NB_WEIGHTS = 0; //value set in init function
int NB_BIASES = 0; // value set in init function


vector<matrix<double>> W(NB_LAYERS); //list of matrices of weights for each layer
vector<vector<double>> B(NB_LAYERS); //list of biases for each layer
//W and B are used from *1* to NB_LAYERS-1 included

vector<matrix<double>> zeroWeights(NB_LAYERS); //initialised in init()
vector<vector<double>> zeroBiases(NB_LAYERS); // initialised in init()

void init()
{   


    std::cout << "\n ## INITIALIZING ## \n";


    for(int layer = 0; layer < NB_LAYERS-1; layer++)
    {
        NB_WEIGHTS += NB_NODES_PER_LAYER[layer] * NB_NODES_PER_LAYER[layer+1];
        NB_BIASES += NB_NODES_PER_LAYER[layer+1];
    }


    for(int layer = 1; layer < NB_LAYERS; layer++)
    {
        int l1 = NB_NODES_PER_LAYER[layer-1];
        int l2 = NB_NODES_PER_LAYER[layer];
        W[layer].resize(l2,l1);
        B[layer].resize(l2);
        zeroWeights[layer].resize(l2,l1);
        zeroBiases[layer].resize(l2);

        for(int j = 0; j < l2; j++)
        {
            for(int k = 0; k < l1; k++)
            {
                W[layer](j,k) = (double)(rand() % 3 - 1) / 10;
                zeroWeights[layer](j,k) = 0;
            }
            B[layer][j] = (double)(rand() % 3 - 1) / 10;
            zeroBiases[layer][j] = 0;
        }
    }

}


void updateGradient(vector<matrix<double>>& gradientW, vector<vector<double>>& gradientB,
const vector<vector<double>>& zValues, const vector<int>& desiredOutput)
{
    vector<matrix<double>> nablaC_W = zeroWeights;
    vector<vector<double>> nablaC_B = zeroBiases;

    vector<double> lastLayerZ = zValues[NB_LAYERS-1];

    vector<double> lastLayerActivation = lastLayerZ;
    apply(sigmoid, lastLayerActivation);

    vector<double> nabla_a_C = 2.0 * (lastLayerActivation - desiredOutput);

    vector<double> fPrime = lastLayerZ;
    apply(sigmoidPrime, fPrime);         // This will quickly saturate to 0 if values are large (amplitude)
    
    vector<double> delta = element_prod(nabla_a_C,  fPrime);

    for(int layer = NB_LAYERS-2; layer >= 0; layer--)
    {
        vector<double> activations = zValues[layer];  
        apply(sigmoid, activations);

        //weights 
        for(int j = 0; j < NB_NODES_PER_LAYER[layer+1]; j++)
        {
            for(int k = 0; k < NB_NODES_PER_LAYER[layer]; k++)
            {
                nablaC_W[layer+1](j,k) = activations[k] * delta[j];
                
            }
        }
        //biases
        for(int j = 0; j < NB_NODES_PER_LAYER[layer+1]; j++)
        {
            nablaC_B[layer+1][j] = delta[j];
        }

        vector<double> z_l = zValues[layer];
        apply(sigmoidPrime, z_l);
        delta = element_prod(prod(trans(W[layer+1]), delta), z_l);

    }

    gradientW += nablaC_W;
    gradientB += nablaC_B;
}

vector<vector<double>> feedForward(int picture_id)
{
    vector<vector<double>> zValues(NB_LAYERS);
    zValues[0].resize(NB_NODES_PER_LAYER[0]);

    for(int i = 0; i < picture_size; i++)
        zValues[0][i] = (double)(pictures[picture_id][i] - 122) / 20;
    

    for(int layer = 1; layer < NB_LAYERS; layer++)
    {
        zValues[layer].resize(NB_NODES_PER_LAYER[layer]);

        vector<double> a = zValues[layer-1];
        apply(sigmoid, a);
        
        zValues[layer] = prod(W[layer], a) + B[layer];
        
    }

    return zValues;
}

vector<int> getDesiredOutput(int picture_id)
{
    vector<int> v(10);
    for(int i = 0; i < 10; i++)
        if(i == labels[picture_id])
            v[i] = 1;
        else
        {
            v[i] = 0;
        }
        
    return v;
}

void updateParams(vector<matrix<double>> &gradientW, vector<vector<double>> &gradientB)
{
    for(int layer = 0; layer < NB_LAYERS; layer++)
    {
        W[layer] += -alpha * gradientW[layer];
        B[layer] += -alpha * gradientB[layer];  
    }      
}

void test();

void backPropagation()
{

    std::cout << "\n ## BACKPROPAGATION ## \n";
    std::vector<int> pictureIds(training_set_size);
    for(int i = 0; i < training_set_size; i++) pictureIds[i] = i;

    for(int epoch = 0; epoch < nbEpochs; epoch++)
    {
       // if(epoch % 10 == 0)
            std::cout << "Epoch " << epoch << '\n';

        //vector<int> randomInts = getRandomInts(batchSize, 0, training_set_size);
        

        std::random_shuffle(pictureIds.begin(), pictureIds.end());
        
        std::vector<int> batch;
        for(int i = 0; i < training_set_size; i++)
        {   

            batch.push_back(pictureIds[i]);

            if((i+1) % batchSize == 0)
            {
                vector<matrix<double>> gradientW = zeroWeights;
                vector<vector<double>> gradientB = zeroBiases;

                for(int pictureId : batch)  
                {
                    //forward pass
                    vector<vector<double>> zValues = feedForward(pictureId);
                    vector<int> desiredOutput =  getDesiredOutput(pictureId);

                    //backward pass
                    updateGradient(gradientW, gradientB, zValues, desiredOutput);
                }
                batch.resize(0);

                updateParams(gradientW, gradientB);
           }
        }

        if(epoch % 10 == 0)
            test();

    }
}

void test() 
{

    std::cout << "\n ## TESTING ## \n";
    double percentage = 0.0;
    int nbCorrect = 0;

    for(int picture_id = 0; picture_id < test_set_size; picture_id++)
    {
        vector<vector<double>> zValues = feedForward(picture_id + training_set_size);
        vector<double> output = zValues[zValues.size()-1];
        int myValue = 0;
        double max = -oo;
        for(int i = 0; i < (int)output.size(); i++)
        {
            if(output[i] > max)
            {
                max = output[i];
                myValue = i;
            }
        }
        vector<int> trueOutput = getDesiredOutput(picture_id + training_set_size);
        if(trueOutput[myValue] == 1)
            nbCorrect++;
 
    }
    percentage = (double) nbCorrect / test_set_size * 100.0;
    std::cout << "Correct " << nbCorrect << '\n' << "Percentage " << percentage << '\n';
}

int main()
{
    srand(time(0));


    init();

    ReadMNIST();

    
    /* for(int i = 0; i < 28; i++)
    {
        for(int j = 0; j< 28; j++)
            std::cout << (int)pictures[2][i*28 + j];
        std::cout << '\n';
    } */

    // std::cout << (int)labels[2]; 
    backPropagation();

    //print(W, "WEIGHTS");
  //  print(B, "BIASES");
    
    print(feedForward(0));
    print(getDesiredOutput(0));

    print(feedForward(1));
    print(getDesiredOutput(1));

    print(feedForward(2));
    print(getDesiredOutput(2)); 


    test();
}