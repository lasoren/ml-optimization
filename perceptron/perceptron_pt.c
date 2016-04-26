//gcc -o perceptron_pt.o perceptron_pt.c utils.c -lrt -lm
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include "utils.h"

#define MAX_ITERS 1000000
#define TEST_CASE 1
#define GIG 1000000000
#define CPG 1.4           // Cycles per GHz -- Adjust to your computer
#define global_X_dim 6
#define DEBUG 0

double global_w[global_X_dim];
char misclassified[10000];
int global_iters = 0;
int x_length;
int global_sum_missed = 0;
char global_countCheck = 0;
char global_not_classified = 1;
pthread_mutex_t weightMutex;
pthread_mutex_t itersMutex;
pthread_mutex_t classifiedMutex;
pthread_mutex_t sumMissedMutex;
pthread_barrier_t iterationBarrier;

/* used to pass parameters to worker threads */
struct thread_data{
  int thread_id;
  data_t* X;
  char* y;
  double eta; 
  int X_length;
  int X_dim;
  int NUM_THREADS;
};

// helper function for multithreaded perceptron
void* perceptron_helper(void* threadarg){
    // pass the thread a struct containing its thread_id, X and y data, length,
    // and dimensions as well as number of threads.
    struct thread_data *my_data;
    my_data = (struct thread_data *) threadarg;
    int taskid = my_data->thread_id;
    data_t* X = my_data->X;
    char* y = my_data->y;
    double eta = my_data->eta;
    int X_length = my_data->X_length;
    int X_dim = my_data->X_dim;
    int NUM_THREADS = my_data->NUM_THREADS;
    int X_length_low = (taskid * X_length)/NUM_THREADS;
    int X_length_high = X_length_low + (X_length/NUM_THREADS); 

#if DEBUG 
    printf("Hi! I am thread %d computing elements %d to %d\n",taskid,
            X_length_low,X_length_high);
#endif

    char not_classified = 0;
    double w = 0;
    double score[X_length/NUM_THREADS];
    int i, j, sum_missed = 0;
    int local_idx;

    memset(score, 0, (X_length/NUM_THREADS)*sizeof(double));

    //master loop
    while(global_not_classified && global_iters <= MAX_ITERS){

        //set a barrier here to make sure no threads escape the loop before others
        pthread_barrier_wait(&iterationBarrier);

        if (taskid == 0) {  // thread 1 updates the global values.
            pthread_mutex_lock(&sumMissedMutex);
            global_sum_missed = 0;
            pthread_mutex_unlock(&sumMissedMutex);

            pthread_mutex_lock(&itersMutex);
            global_iters++;
            pthread_mutex_unlock(&itersMutex);
        
            pthread_mutex_lock(&classifiedMutex);
            global_not_classified = 0;
            pthread_mutex_unlock(&classifiedMutex);
        }
        
        //set a barrier here to make sure no threads escape the loop before others
        pthread_barrier_wait(&iterationBarrier);

        // each thread updates one index of the weight vector.
        for(i = 0; i < X_length; ++i){
            if(misclassified[i] == 1){
                w += eta*X[i*(X_dim) + taskid]*y[i];
            }
        }
        // each thread updates global weight vector.
        pthread_mutex_lock(&weightMutex);
        global_w[taskid] += w;
        pthread_mutex_lock(&weightMutex);
        
        // global w updated by all threads. Ready to compute misclassified data.
        pthread_barrier_wait(&iterationBarrier);
        
        // each thread classifies 1/NUM_THREADS of the data.
        sum_missed = 0;
        local_idx = 0;
        for (i = X_length_low; i<X_length_high; ++i){
            score[local_idx] = 0;
            for(j = 0; j < X_dim; ++j){
                score[local_idx] += X[i*(X_dim) + j]*global_w[j];
            }

            misclassified[i] = score[local_idx]*y[i] <= 0.0 ? 1 : 0;
            // Set not_classified to 1 if any data point is misclassfied
            // and count number of missed.
            if (misclassified[i] == 1) {
                sum_missed++;
                not_classified = 1;
            }
        }

        pthread_mutex_lock(&classifiedMutex);
        if(not_classified) {
            global_not_classified = 1;
        }
        pthread_mutex_unlock(&classifiedMutex);
        
        pthread_mutex_lock(&sumMissedMutex);
        global_sum_missed += sum_missed;
        pthread_mutex_unlock(&sumMissedMutex);

        //must place barrier here because in our next segment we check
        //the value of global_sum_missed to verify our seperated data. 
        pthread_barrier_wait(&iterationBarrier);
    }

    //verify our seperated data, countCheck should be equal to our NUM_THREADS 
    pthread_mutex_lock(&sumMissedMutex);
    if (global_sum_missed == 0) {
        global_countCheck++;
        // printf("Perfectly separated data\n");
        // printf("sum_missed local = %d\n",sum_missed);
    } 
    else {
        // printf("Finished MAX_ITERS and still %d misclassified\n", sum_missed);
    }
    pthread_mutex_unlock(&sumMissedMutex);
}

//sets up our pthread environment for multithreaded perceptron
void train_perceptron_pt(data_t* X, char* y, double eta, int X_length, int X_dim, int NUM_THREADS){
    //initialization
    long int i, j, k;
    pthread_t threads[NUM_THREADS];
    struct thread_data thread_data_array[NUM_THREADS];
    int rc;
    long t;
    memset(misclassified, 1, (X_length)*sizeof(char));

    for (t = 0; t < NUM_THREADS; t++) {
        thread_data_array[t].thread_id = t;
        thread_data_array[t].X = X;
        thread_data_array[t].y = y;
        thread_data_array[t].eta = eta;
        thread_data_array[t].X_length = X_length;
        thread_data_array[t].X_dim = X_dim;
        thread_data_array[t].NUM_THREADS = NUM_THREADS;
        rc = pthread_create(&threads[t], NULL, perceptron_helper,
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

int main(int argc, const char** argv){ 
    
    //initialization
    struct timespec diff(struct timespec start, struct timespec end);
    struct timespec time1, time2, difference;
    int X_length = 10000;
    int X_dim = 6;
    int test_case = TEST_CASE;
    data_t X[X_length*X_dim];
    char y[X_length];
    long int i, j, k;
    long int time_sec, time_ns;
    int NUM_THREADS = 6;
	float eta;

    memset(global_w, 0, (X_dim)*sizeof(double));
    printf("\n Hello World -- Perceptron multithreaded\n");
    
    //parse the input file
    int line_counter = 0;
    FILE* stream = fopen("data.csv", "r");
    char line[1024];
    while (fgets(line, 1024, stream))
    {
        char* tmp = strdup(line);
        int idx = line_counter*X_dim;
        X[idx] = 1.0;
        X[idx + 1] = strtod(getfield(tmp, 1), NULL);
        tmp = strdup(line);
        X[idx + 2] = strtod(getfield(tmp, 2), NULL);
        X[idx + 3] = X[idx + 1]*X[idx + 2]; // xy
        X[idx + 4] = X[idx + 1]*X[idx + 1]; // x^2 
        X[idx + 5] = X[idx + 2]*X[idx + 2]; // y^2
        // NOTE strtok clobbers tmp
        free(tmp);
        line_counter++;
    }

    assign_labels(X, X_length, X_dim, test_case, y);

    //display values
    for (i = 0; i < X_length; i++) {
        for (j = 0; j < X_dim; j++) {
            printf("%f,",X[i*X_dim+j]);
        }
        printf("%d",y[i]);
        printf("\n");
    }

    //initialize pthread utlities
    if (pthread_mutex_init(&weightMutex, NULL) != 0){
        printf("\n weight mutex init failed\n");
        return 1;
    }
    if (pthread_mutex_init(&itersMutex, NULL) != 0){
        printf("\n iters mutex init failed\n");
        return 1;
    }
    if (pthread_mutex_init(&classifiedMutex, NULL) != 0){
        printf("\n classified mutex init failed\n");
        return 1;
    }
    if (pthread_mutex_init(&sumMissedMutex, NULL) != 0){
        printf("\n sum missed mutex init failed\n");
        return 1;
    }
    if (pthread_barrier_init(&iterationBarrier, NULL, NUM_THREADS) != 0){
        printf("\n sum missed mutex init failed\n");
        return 1;
    }

    //time the multithreaded perceptron function
    i=0;	
    printf("size, running time, num iters\n");
    for (x_length = 600; x_length <= X_length; x_length += 600) {
        for(i = 0; i < 5; i++){
            clock_gettime(CLOCK_REALTIME, &time1);
            train_perceptron_pt(X, y, 1.0, x_length, X_dim, NUM_THREADS);
            clock_gettime(CLOCK_REALTIME, &time2);
            difference = diff(time1,time2);
            printf("%d, %f, %d\n",
                    x_length,
                    (double) (GIG * difference.tv_sec + difference.tv_nsec),
                    global_iters);
            memset(global_w, 0, (X_dim)*sizeof(double));
            global_iters = 0;
            global_not_classified = 1;
        }
    }

    pthread_mutex_destroy(&weightMutex);
    pthread_mutex_destroy(&itersMutex);
    pthread_mutex_destroy(&classifiedMutex);
    pthread_mutex_destroy(&sumMissedMutex);
    pthread_barrier_destroy(&iterationBarrier);
    
    printf("\n");

    return 0;
}
