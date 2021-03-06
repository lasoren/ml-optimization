// TO COMPILE
// g++ -std=c++0x -o knn_pt.o main_pt.cc knn_pt.cc knn.cc -lrt -lm

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <iostream>

#include "knn_pt.h"

#define GIG 1000000000
using namespace std;

const char* getfield(char* line, int num);

int main() {
    //initialization
    struct timespec diff(struct timespec start, struct timespec end);
    struct timespec time1, time2, difference;
    const int x_length = 500;
    const int x_dim = 2;
    int i, j;

    const int labeled_length = 200;

    data_t x[x_length*x_dim];
    data_t labeled[labeled_length*x_dim];
    data_t x_labels[x_length];
    data_t x_pred[x_length];
    data_t labels[labeled_length];

    int line_counter = 0;

    FILE* stream = fopen("train2d.csv", "r");
    char line[1024];
    while (fgets(line, 1024, stream))
    {
        char* tmp = strdup(line);
        int idx = line_counter*x_dim;
        labeled[idx] = strtod(getfield(tmp, 1), NULL);
        tmp = strdup(line);
        labeled[idx + 1] = strtod(getfield(tmp, 2), NULL);
        tmp = strdup(line);
        labels[line_counter] = strtod(getfield(tmp, 3), NULL);
        free(tmp);
        line_counter++;
    }

    stream = fopen("test2d.csv", "r");
    line_counter = 0;
    while (fgets(line, 1024, stream))
    {
        char* tmp = strdup(line);
        int idx = line_counter*x_dim;
        x[idx] = strtod(getfield(tmp, 1), NULL);
        tmp = strdup(line);
        x[idx + 1] = strtod(getfield(tmp, 2), NULL);
        tmp = strdup(line);
        x_labels[line_counter] = strtod(getfield(tmp, 3), NULL);
        free(tmp);
        line_counter++;
    }
/*
    for (i = 0; i < x_length; i++) {
        for (j = 0; j < x_dim; j++) {
            printf("%f,", x[i*x_dim+j]);
        }
        printf("%d",x_labels[i]);
        printf("\n");
    }
*/  clock_gettime(CLOCK_REALTIME, &time1);
    perform_knn_pt(6, x, labeled, labels, x_dim, x_length, labeled_length, x_pred);
    clock_gettime(CLOCK_REALTIME, &time2);
    difference = diff(time1,time2);
	printf("size, running time\n");
    printf("%d, %f\n",
    x_length,(double) (GIG * difference.tv_sec + difference.tv_nsec));
    int count_right = 0;    
    for (i = 0; i < x_length; i++) {
        if (x_pred[i] == x_labels[i]) {
            count_right++;
        }
    } 
    cout << "Percentage correctly classified: " <<
        100*count_right / (float) x_length << endl;
    return 0;
}

const char* getfield(char* line, int num) {
    const char* tok;
    for (tok = strtok(line, ",");
            tok && *tok;
            tok = strtok(NULL, ",\n"))
    {
        if (!--num)
            return tok;
    }
    return NULL;
}

