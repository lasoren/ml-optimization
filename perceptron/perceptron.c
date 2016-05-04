// gcc -o perceptron.o perceptron.c utils.c -lrt -lm

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "utils.h"

#define GIG 1000000000
#define MAX_ITERS 10000
#define TEST_CASE 1
#define DEBUG 1
#define TIMING 0

double global_w[23];

int train_perceptron(data_t* x, char* y, double eta, int x_length, int x_dim) {
    double w[x_dim];
    double score[x_length];
    char misclassified[x_length];
    char not_classified = 1;
    int i, j, sum_missed, iters = 0;
    int min_missed = -1;
    int iters_since_best = 0;
    double w_converge[x_dim];
    //set w to 0's, misclassified to 1's
    memset(w, 0, x_dim*sizeof(double));
    memset(misclassified, 1, x_length*sizeof(char));
    while(not_classified && iters <= MAX_ITERS){
        iters++;
        not_classified = 0;
        for(i=0; i < x_length; ++i){
            if(misclassified[i] == 1){
                for(j=0; j< x_dim; ++j){
                    w[j] = w[j] + eta*x[i*x_dim + j]*y[i];
                }
            }
        }
        sum_missed = 0;
        for (i=0; i<x_length; ++i) {
            score[i] = 0;
            for (j = 0; j < x_dim; j++) {
                score[i] += x[i*x_dim + j]*w[j];
            }
            misclassified[i] = score[i]*y[i] <= 0.0 ? 1 : 0;
            // Set not_classified to 1 if any data point is misclassfied
            // and count number of missed.
            if (misclassified[i] == 1) {
                sum_missed++;
                not_classified = 1;
            }
        }
        if (min_missed == -1 || sum_missed < min_missed) {
            min_missed = sum_missed;
            iters_since_best = 0;
            for (j = 0; j < x_dim; j++) {
                w_converge[j] = w[j];
            }
        } else {
            if (iters_since_best >= 10) {
                for (j = 0; j < x_dim; j++) {
                    w[j] = w_converge[j];
                    break;
                }
            }
            iters_since_best++;
        }
#if DEBUG
        if (iters%1000 == 0)
            printf("Number of iterations so far: %d\n", iters);
#endif
    }
#if DEBUG
    if (sum_missed == 0) {
        printf("Perfectly separated data\n");
    } else {
        printf("Finished MAX_ITERS and still %d misclassified\n", sum_missed);
    }
#endif
    // Set into the global variable so I can access in main.
    for (j = 0; j < x_dim; j++) {
        global_w[j] = w[j];
        break;
    }
    return iters;
}

int main(int argc, const char** argv){ 
    struct timespec time1, time2, difference;
    // Training data.
    const int x_test_length = 25000;
    const int x_dim = 23;
    data_t* x_test = (data_t*) malloc(x_test_length*x_dim*sizeof(data_t));
    char y[x_test_length];
    // Testing data.
    const int x_length = 5000;
    data_t* x = (data_t*) malloc(x_length*x_dim*sizeof(data_t));
    char y_t[x_length];

    int i, j;
    float eta;

    int line_counter = 0;
    FILE* stream = fopen("../datasets/percept-credit-card-clients.csv", "r");
    int write_to = 0;
    char line[8192];
    while (fgets(line, 8192, stream))
    {
        if (write_to == 0) {
            char* tmp = strdup(line);
            int idx = line_counter*x_dim;
            // Get the 23 x dimensions.
            get_fields(tmp, x_test+idx, x_dim);
            // Get the 1 Y dimension from the dataset.
            tmp = strdup(line);
            y[line_counter] = (char) strtod(getfield(tmp, 24), NULL);
            free(tmp);
            line_counter++;
            if (line_counter == x_test_length) {
                write_to = 1;
                line_counter = 0;
            }
        } else {
            char* tmp = strdup(line);
            int idx = line_counter*x_dim;
            get_fields(tmp, x+idx, x_dim);
            tmp = strdup(line);
            y_t[line_counter] = (data_t) strtod(getfield(tmp, 24), NULL);

            free(tmp);
            line_counter++;
            if (line_counter == x_test_length + x_length) {
                break;
            }
        }
    }
    
    for (i = 0; i < 50; i++) {
        for (j = 0; j < x_dim; j++) {
            printf("%f,", x_test[i*x_dim+j]);
        }
        printf("%d",y[i]);
        printf("\n");
    }

    clock_gettime(CLOCK_REALTIME, &time1);
    int iterations = train_perceptron(x_test, y, 0.1, x_test_length, x_dim);
    clock_gettime(CLOCK_REALTIME, &time2);
    difference = diff(time1,time2);
    printf("%d, %f, %d\n", x_test_length, (double) (GIG * difference.tv_sec + difference.tv_nsec),iterations);
    printf("Number of iterations: %d\n", iterations); 

    double score = 0;
    int count_correct = 0;
    for (i = 0; i < x_length; i++) {
        score = 0;
        for (j = 0; j < x_dim; j++) {
            score += x[i*x_dim + j] * global_w[j];
        }
        if (score*y_t[i] > 0) {
            count_correct++;
        }
    }
    printf("%d correct out of %d\n", count_correct, x_length);

    return 0;
}

