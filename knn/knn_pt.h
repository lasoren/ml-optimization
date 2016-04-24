
#include <vector>
#include <algorithm>

typedef double data_t;

using namespace std;

/* used to pass parameters to worker threads */
struct thread_data{
  int thread_id;
  int k;
  data_t* x;
  data_t* labeled;
  data_t* labels;
  int dim;
  int x_length;
  int labeled_length;
  data_t* x_pred;
  int NUM_THREADS;
};

// Helper function for multithreaded KNN
void* knn_helper(void* threadarg);

// Perform K nearest neighbor classification on all x data points.
void perform_knn_pt(int k,
         data_t* x,
         data_t* labeled,
         data_t* labels,
         int dim,
         int x_length,
         int labeled_length,
         data_t* x_pred);

