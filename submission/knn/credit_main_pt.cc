// g++ -o knn.o credit_main_pt.cc knn_pt.cc knn.cc utils.cc -lrt -lm

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <iostream>

#include "knn_pt.h"
#include "utils.h"

#define DEBUG 0
#define GIG 1000000000

using namespace std;

// Get fields from a line and fill an array with the values.
void get_fields(char* line, data_t* data, int max_width);

const char* getfield(char* line, int num);

int main() {
    struct timespec time1, time2, difference;
    const int x_length = 5000;
    const int labeled_length = 25000;
    const int x_dim = 23;
    int i, j;

    data_t x[x_length*x_dim];
    data_t* labeled = (data_t*) malloc(labeled_length*x_dim*sizeof(data_t));
    data_t x_labels[x_length];
    data_t x_pred[x_length];
    data_t labels[labeled_length];

    int line_counter = 0;
    FILE* stream = fopen("../datasets/percept-credit-card-clients.csv", "r");
    char line[10240];
    int write_to = 0;
    while (fgets(line, 10240, stream))
    {
        if (write_to == 0) {
            char* tmp = strdup(line);
            int idx = line_counter*x_dim;
            get_fields(tmp, labeled+idx, x_dim);
            tmp = strdup(line);
            labels[line_counter] = (data_t) strtod(getfield(tmp, 24), NULL);

            free(tmp);
            line_counter++;
            if (line_counter == labeled_length) {
                write_to = 1;
                line_counter = 0;
            }
        } else {
            char* tmp = strdup(line);
            int idx = line_counter*x_dim;
            get_fields(tmp, x+idx, x_dim);
            tmp = strdup(line);
            x_labels[line_counter] = (data_t) strtod(getfield(tmp, 24), NULL);

            free(tmp);
            line_counter++;
            if (line_counter == x_length + labeled_length) {
                break;
            }
        }
    }

    for (i = 0; i < 50; i++) {
        for (j = 0; j < x_dim; j++) {
            printf("%f,", labeled[i*x_dim+j]);
        }
        printf("%f",labels[i]);
        printf("\n");
    }
    for (i = 0; i < 50; i++) {
        for (j = 0; j < x_dim; j++) {
            printf("%f,", x[i*x_dim+j]);
        }
        printf("%f",x_labels[i]);
        printf("\n");
    }

    cout << "k, running time, percentage right\n" << endl;
    for (int k = 1; k <= 10; k++) { 
        clock_gettime(CLOCK_REALTIME, &time1);
        perform_knn_pt(k, x, labeled, labels, x_dim, x_length, labeled_length, x_pred);
        clock_gettime(CLOCK_REALTIME, &time2);
        difference = diff(time1,time2);
        int count_right = 0;    
        for (i = 0; i < x_length; i++) {
            if (x_pred[i] == x_labels[i]) {
                count_right++;
            }
        } 
        cout << k << ", " << 
            (double) (GIG * difference.tv_sec + difference.tv_nsec) << ", " <<
            100*count_right / (float) x_length << endl;
    }
    free(labeled);
    return 0;
}

