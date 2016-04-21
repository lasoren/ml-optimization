
#include <vector>
#include <algorithm>

typedef char data_t;

using namespace std;

// Perform K nearest neighbor classification on all x data points.
void perform_knn(const int k,
         const data_t* x,
         const data_t* labeled,
         const data_t* labels,
         const int dim,
         const int x_length,
         const int labeled_length,
         data_t* x_pred);

// Perform K nearest neighbor classification on all one data point at index i.
void knn(const int k,
         const data_t* x,
         const int i,
         const data_t* labeled,
         const data_t* labels,
         const int dim,
         const int labeled_length,
         data_t* x_pred);

// Compute the euclidean distance between two multi-dimensional vectors.
// Note: x and y must be the same length.
float euclid_distance(const data_t* x, const data_t* y, const int length);

// Computes the hamming distance between any multi-dimensional vectors where
// where the difference between each dimension determines the hamming distance.
data_t hamming_distance(const data_t* x, const data_t* y, const int length);

// Given the distances from this test point to all of the labeled points
// and the number of nearest neighbors to consider, predict this point's
// label.
int predict_class(const float* distances,
                  const data_t* labels,
                  const int length,
                  const int k);

// Compute the mode of the nearest neighbor labels.
int mode_labels(const vector< pair<int, float> >& labels);

// Used for sorting pairs of integer and float values.
bool compare_labels(const pair<int, float>&i, const pair<int, float>&j);


