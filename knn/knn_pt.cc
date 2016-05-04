#include "knn_pt.h"

#include <math.h>
#include <map>
#include <utility>
#include <functional>
#include <pthread.h>
#include <cstdio>
#include <numeric>
#include <iostream>

#include "knn.h"

using namespace std;

void* knn_helper(void* threadarg){
    struct thread_data *my_data;
    my_data = (struct thread_data *) threadarg;
    int taskid = my_data->thread_id;
    int k = my_data->k;
    data_t* x = my_data->x;
    data_t* labeled = my_data->labeled;
    data_t* labels = my_data->labels;
    int dim = my_data->dim;
    int x_length = my_data->x_length;
    int labeled_length = my_data->labeled_length;
    data_t* x_pred = my_data->x_pred;
    int NUM_THREADS = my_data->NUM_THREADS;
    int x_length_low = (taskid * x_length)/NUM_THREADS;
    int x_length_high = x_length_low + (x_length/NUM_THREADS); 
 
#if DEBUG
    printf("Hi! I am thread %d computing elements %d to %d\n",
            taskid,x_length_low,x_length_high);
#endif

    // Loop through data points and classify each based on nearest labeled
    // neighbors.
    for (int i = x_length_low; i < x_length_high; i++) {
#if DEBUG
        if (i%100 == 0) {
            cout << "Done classifying: " << i << endl;
        }
#endif
        knn(k, x, i, labeled, labels, dim, labeled_length, x_pred);
    }
}

void perform_knn_pt(int k,
         data_t* x,
         data_t* labeled,
         data_t* labels,
         int dim,
         int x_length,
         int labeled_length,
         data_t* x_pred) {
    //initialization
    int NUM_THREADS = 10;
    pthread_t threads[NUM_THREADS];
    struct thread_data thread_data_array[NUM_THREADS];
    int rc;
    long t;

    for (t = 0; t < NUM_THREADS; t++) {
        thread_data_array[t].thread_id = t;
        thread_data_array[t].k = k;
        thread_data_array[t].x = x;
        thread_data_array[t].labeled = labeled;
        thread_data_array[t].labels = labels;
        thread_data_array[t].dim = dim;
        thread_data_array[t].x_length = x_length;
        thread_data_array[t].labeled_length = labeled_length;
        thread_data_array[t].x_pred = x_pred;
        thread_data_array[t].NUM_THREADS = NUM_THREADS;
        rc = pthread_create(&threads[t], NULL, knn_helper,
                (void*) &thread_data_array[t]);
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    for (t = 0; t < NUM_THREADS; t++) {
        if (pthread_join(threads[t],NULL)){ 
            printf("ERROR; code on return from join is %d\n", rc);
            exit(-1);
        }
    }
}

