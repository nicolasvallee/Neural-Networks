#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>
#include <cmath>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace boost::numeric::ublas;

const int NB_LAYERS = 3; //Number of layers of nodes, including the input and output layer

const int picture_size = 28 * 28;
const int NB_NODES_PER_LAYER[NB_LAYERS] = 
{picture_size, 10, 10};

int NB_WEIGHTS = 0; //value set in init function
int NB_BIASES = 0; // value set in init function

const int training_set_size = 3;
const int test_set_size = 0;
const double oo = 1e8;

vector<vector<uint8_t>> pictures(training_set_size + test_set_size);
int iPicture = 0;
vector<uint8_t> labels(training_set_size + test_set_size);
int iLabel = 0;


int32_t ReverseInt (int32_t i)
{
    unsigned char ch1, ch2, ch3, ch4;  //gives you at least the 0-255 range
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int32_t)ch1<<24)+((int32_t)ch2<<16)+((int32_t)ch3<<8)+ch4;
}

void readImages(std::ifstream& f, int nbImages)
{
    int32_t magic_number=0;
    int32_t number_of_images=0;
    int32_t n_rows=0;
    int32_t n_cols=0;
    f.read((char*)&magic_number,sizeof(magic_number));
    magic_number= ReverseInt(magic_number);
    f.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images= ReverseInt(number_of_images);
    f.read((char*)&n_rows,sizeof(n_rows));
    n_rows= ReverseInt(n_rows);
    f.read((char*)&n_cols,sizeof(n_cols));
    n_cols= ReverseInt(n_cols);
    

    

    for(;iPicture< nbImages;++iPicture)
    {
        pictures[iPicture].resize(picture_size);

        for(int r=0;r<n_rows;++r)
        {
            for(int c=0;c<n_cols;++c)
            {
                unsigned char temp=0;
                f.read((char*)&temp,sizeof(temp));
                pictures[iPicture][(n_rows*r)+c]= (uint8_t)temp;
            }
        }
    }

}

void readLabels(std::ifstream& f, int nbLabels)
{

        int32_t magic_number=0;
        int32_t number_of_labels=0;
        f.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        f.read((char*)&number_of_labels,sizeof(number_of_labels));
        number_of_labels= ReverseInt(number_of_labels);
    
        for(;iLabel<nbLabels;++iLabel)
        {
            unsigned char temp = 0;
            f.read((char*)&temp, sizeof(temp));
            labels[iLabel] = (uint8_t)temp;
        }

}

void ReadMNIST()
{

    std::cout << "\n ## READING FILES ## \n";
    std::ifstream imagesTraining("train-images-idx3-ubyte", std::ios::binary);
    std::ifstream labelsTraining("train-labels-idx1-ubyte", std::ios::binary);

    std::ifstream imagesTest("t10k-images-idx3-ubyte", std::ios::binary);
    std::ifstream labelsTest("t10k-labels-idx1-ubyte", std::ios::binary);

    if (imagesTraining.is_open() && labelsTraining.is_open() && imagesTest.is_open() && labelsTest.is_open())
    {
        readImages(imagesTraining, training_set_size);
        readImages(imagesTest, training_set_size + test_set_size);

        readLabels(labelsTraining, training_set_size);
        readLabels(labelsTest, training_set_size +  test_set_size);

    }
    else
    {
        std::cout << "ERROR: file not open";
    }
    
}



vector<matrix<double>> W; //list of matrices of weights for each layer
vector<vector<double>> B; //list of biases for each layer


const double steepness = 1.0;

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double sigmoidPrime(double x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}

double RELU(double x)
{
    return std::max(0.0, x);
}

double RELUPrime(double x)
{
    if(x <= 0.0)
        return 0;
    else
        return 1;
    
}


void apply(double func(double), vector<double> &v)
{
    for(double &a : v)
        a = func(a);
}

template <typename T>
void print(const T &t, std::string msg = "")
{
    std::cout << msg << "\n\n" << t << "\n\n";
}

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

vector<int> getRandomInts(int n, int a, int b)
{
    bool seen[b-a];
    std::fill(seen,seen + b - a, false);

    vector<int> v(n);
    if(n > b - a)
    {
        std::cout << "ERROR in getRandomInts \n";
        exit(-1);
    }
    else
    {
        for(int i = 0; i < n; i++)
        {
            int x;
            do
            {
                x = a + rand() % (b - a);
            }
            while(seen[x-a]);
            seen[x-a] = true;
            v[i] = x;
        }
    }
    
    return v;
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

void backPropagation()
{

    std::cout << "\n ## BACKPROPAGATION ## \n";
    int batchSize = std::min(training_set_size, 10);
    int nbTrainings = 100;

    for(int iTrain = 0; iTrain < nbTrainings; iTrain++)
    {
        if(iTrain % 100 == 0)
            std::cout << "Training " << iTrain << '\n';

        vector<int> randomInts = getRandomInts(batchSize, 0, training_set_size);

        vector<double> gradient(NB_WEIGHTS + NB_BIASES);
        std::fill(gradient.begin(), gradient.end(), 0);
        
        for(int picture_id: randomInts)
        {
            vector<vector<double>> zValues = computePerceptron(picture_id);
            vector<int> desiredOutput =  getDesiredOutput(picture_id);

            updateGradient(gradient, zValues, desiredOutput);
        }


        //std::cout << "gradient" << gradient << "\n";
        double alpha = pow(10,0);
        //std::cout << alpha << '\n';
        vector<double> dG = -alpha * gradient;
       
        updateParams(dG);
        

    }
}

void test() 
{

    std::cout << "\n ## TESTING ## \n";
    double percentage = 0.0;
    int nbCorrect = 100;

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
        vector<int> trueOutput = getDesiredOutput(picture_id);
        if(trueOutput[myValue] == 1)
            nbCorrect++;
 
    }
    percentage = (double) nbCorrect / test_set_size;
    std::cout << nbCorrect << '\n' << percentage << '\n';
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

    print(W, "WEIGHTS");
    print(B, "BIASES");
    
    print(computePerceptron(0));
    print(getDesiredOutput(0));

    print(computePerceptron(1));
    print(getDesiredOutput(1));

    print(computePerceptron(2));
    print(getDesiredOutput(2)); 


    //test();
}