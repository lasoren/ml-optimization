#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "utils.h"

#define GIG 1000000000
#define MAX_ITERS 100000
#define TEST_CASE 1

void train_perceptron(data_t* x, char* y, double eta, int x_length, int x_dim){
    double w[x_dim];
    double score[x_length];
    char misclassified[x_length];
    char not_classified = 1;
    int i, j, sum_missed, iters = 0;
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
    }
    if (sum_missed == 0) {
        printf("Perfectly separated data\n");
    } else {
        printf("Finished MAX_ITERS and still %d misclassified\n", sum_missed);
    }
}

struct timespec diff(struct timespec start, struct timespec end){
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}

int main(int argc, const char** argv){ 
    struct timespec diff(struct timespec start, struct timespec end);
    struct timespec time1, time2, difference;
	struct timespec differences[19];
    const int X_length = 5000;
    const int X_dim = 6;
    data_t X[X_length*X_dim];
    char y[X_length];
    int i, j;
	float eta;

    int test_case = TEST_CASE;
    
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

    for (i = 0; i < X_length; i++) {
        for (j = 0; j < X_dim; j++) {
            printf("%f,", X[i*X_dim+j]);
        }
        printf("%d",y[i]);
        printf("\n");
    }

	i=0;	
	for(eta = 0.1; eta <= 1.0; eta+= .05){
    	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
	    train_perceptron(X, y, eta, X_length, X_dim);
    	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    	difference = diff(time1,time2);
    	printf("Eta:%f, Total time in ns: %f\n",eta,(double)
    	(GIG * difference.tv_sec + difference.tv_nsec));
	}
    return 0;
}
