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


vector<matrix<double>> W; //list of matrices of weights for each layer
vector<vector<double>> B; //list of biases for each layer



void init()
{   


    std::cout << "\n ## INITIALIZING ## \n";


    for(int layer = 0; layer < NB_LAYERS-1; layer++)
    {
        NB_WEIGHTS += NB_NODES_PER_LAYER[layer] * NB_NODES_PER_LAYER[layer+1];
        NB_BIASES += NB_NODES_PER_LAYER[layer+1];
    }

    W.resize(NB_LAYERS); //W and B are used from *1* to NB_LAYERS-1 included
    B.resize(NB_LAYERS);  

    for(int layer = 1; layer < NB_LAYERS; layer++)
    {
        int l1 = NB_NODES_PER_LAYER[layer-1];
        int l2 = NB_NODES_PER_LAYER[layer];
        W[layer].resize(l2,l1); // is it not(l1,l2) ?????????
        B[layer].resize(l2);
         for(int j = 0; j < l2; j++)
        {

            for(int k = 0; k < l1; k++)
            {
                W[layer](j,k) = (double)(rand() % 3 - 1) / 10;
                //W[layer](j,k) = 0;
            }
            B[layer][j] = (double)(rand() % 3 - 1) / 10;
            //B[layer+1][j] = 0;
        }
    }

}


void updateGradient(vector<double>& gradient, const vector<vector<double>>& zValues, const vector<int>& desiredOutput)
{
    vector<double> nablaC(NB_WEIGHTS + NB_BIASES);
    //Ordered by decreasing number of layer:
    // weights first :
    //then by increasing nodes on the right (j)
    //then by increasing nodes on the left (k)
    //then biases

    vector<double> lastLayerZ = zValues[NB_LAYERS-1];

  //  std::cout << "lastLayerZ" << lastLayerZ<< "\n"; 
    vector<double> lastLayerActivation = lastLayerZ; // THIS WILL CONSTRUCT A NEW OBJECT RIGHT ??
    apply(sigmoid, lastLayerActivation);

   // std::cout << "lastLayerA" << lastLayerActivation << "\n"; 
    vector<double> nabla_a_C = 2.0 * (lastLayerActivation - desiredOutput);

   // std::cout << "nabla_a_C" << nabla_a_C << "\n"; 
    int i_nablaC = 0;
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
                nablaC[i_nablaC] = activations[k] * delta[j];
                i_nablaC++;
            }
        }
        //biases
        for(int j = 0; j < NB_NODES_PER_LAYER[layer+1]; j++)
        {
            nablaC[i_nablaC] = delta[j];
            i_nablaC++;
        }
        vector<double> v1 = prod(trans(W[layer+1]), delta);
        vector<double> v2 = zValues[layer];
        apply(sigmoidPrime, v2);
        delta = element_prod(v1, v2);

    }

    gradient += nablaC;
}

vector<vector<double>> computePerceptron(int picture_id)
{
    vector<vector<double>> zValues(NB_LAYERS);
    zValues[0].resize(NB_NODES_PER_LAYER[0]);

    for(int i = 0; i < picture_size; i++)
        zValues[0][i] = (double)(pictures[picture_id][i] - 122) / 20;
    
    

    for(int layer = 1; layer < NB_LAYERS; layer++)
    {
        zValues[layer].resize(NB_NODES_PER_LAYER[layer]);

        for(int j = 0; j < NB_NODES_PER_LAYER[layer]; j++)
        {
            double z = 0;
            for(int k = 0; k < NB_NODES_PER_LAYER[layer-1]; k++)
            {   
                double activation = sigmoid(zValues[layer-1][k]);
                z +=  activation * W[layer](j,k);
            }
                

            z += B[layer][j];
            zValues[layer][j] = z; 
        }
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

void updateParams(vector<double> &dG)
{

    int i = 0;
    for(int layer = NB_LAYERS-1; layer >= 1; layer--)

    {
        for(int j = 0; j < NB_NODES_PER_LAYER[layer]; j++)
        {
            for(int k = 0; k < NB_NODES_PER_LAYER[layer-1]; k++)
            {
                W[layer](j,k) += dG[i];
                i++;
            }
        }
        for(int j = 0; j < NB_NODES_PER_LAYER[layer]; j++)
        {
            B[layer][j] += dG[i];
            i++;
        }   
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
        if(epoch % 10 == 0)
            std::cout << "Epoch " << epoch << '\n';

        //vector<int> randomInts = getRandomInts(batchSize, 0, training_set_size);
        

        std::random_shuffle(pictureIds.begin(), pictureIds.end());
        
        std::vector<int> batch;
        for(int i = 0; i < training_set_size; i++)
        {   

            batch.push_back(pictureIds[i]);

            if((i+1) % batchSize == 0)
            {
                vector<double> gradient(NB_WEIGHTS + NB_BIASES);
                std::fill(gradient.begin(), gradient.end(), 0);

                for(int pictureId : batch)  
                {
                    //forward pass
                    vector<vector<double>> zValues = computePerceptron(pictureId);
                    vector<int> desiredOutput =  getDesiredOutput(pictureId);

                    //backward pass
                    updateGradient(gradient, zValues, desiredOutput);
                }
                batch.resize(0);

                //std::cout << "gradient" << gradient << "\n";
                //std::cout << alpha << '\n';
                vector<double> dG = -alpha * gradient;
       
                updateParams(dG);
           }
        }

       // test();

    }
}

void test() 
{

    std::cout << "\n ## TESTING ## \n";
    double percentage = 0.0;
    int nbCorrect = 0;

    for(int picture_id = 0; picture_id < test_set_size; picture_id++)
    {
        vector<vector<double>> zValues = computePerceptron(picture_id + training_set_size);
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
    
    print(computePerceptron(0));
    print(getDesiredOutput(0));

    print(computePerceptron(1));
    print(getDesiredOutput(1));

    print(computePerceptron(2));
    print(getDesiredOutput(2)); 


    test();
}