#ifndef UTILS_H
#define UTILS_H


#include <boost/numeric/ublas/vector.hpp>
#include "constants.hpp"

using namespace boost::numeric::ublas;

int iPicture = 0;
int iLabel = 0;
vector<vector<uint8_t>> pictures(training_set_size + test_set_size);

vector<uint8_t> labels(training_set_size + test_set_size);

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
#endif