#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <iostream>

#include "knn.h"

using namespace std;

typedef char data_t;

// Get fields from a line and fill an array with the values.
void get_fields(char* line, data_t* data);

int main() {
    const int x_length = 2163;
    const int labeled_length = 13007;
    const int x_dim = 784;
    int i, j;

    data_t x[x_length*x_dim];
    data_t* labeled = (char*) malloc(labeled_length*x_dim*sizeof(data_t));
    data_t x_labels[x_length];
    data_t x_pred[x_length];
    data_t labels[labeled_length];

    int line_counter = 0;
    FILE* stream = fopen("Xtrain.csv", "r");
    char line[10240];
    while (fgets(line, 1024, stream))
    {
        char* tmp = strdup(line);
        int idx = line_counter*x_dim;
        get_fields(tmp, labeled+idx);
        free(tmp);
        cout << line_counter << ",";
        line_counter++;
    }

    stream = fopen("ytrain.csv", "r");
    line_counter = 0;
    while (fgets(line, 1024, stream))
    {
        char* tmp = strdup(line);
        get_fields(tmp, labels+line_counter);
        free(tmp);
        cout << line_counter << ",";
        line_counter++;
    }
    
    stream = fopen("Xtest.csv", "r");
    line_counter = 0;
    while (fgets(line, 1024, stream))
    {
        char* tmp = strdup(line);
        int idx = line_counter*x_dim;
        get_fields(tmp, x+idx);
        free(tmp);
        cout << line_counter << ",";
        line_counter++;
    }

    stream = fopen("ytest.csv", "r");
    line_counter = 0;
    while (fgets(line, 1024, stream))
    {
        char* tmp = strdup(line);
        int idx = line_counter*x_dim;
        get_fields(tmp, x_pred+idx);
        free(tmp);
        cout << line_counter << ",";
        line_counter++;
    }

    perform_knn(6, x, labeled, labels, x_dim, x_length, labeled_length, x_pred);
    int count_right = 0;    
    for (i = 0; i < x_length; i++) {
        if (x_pred[i] == x_labels[i]) {
            count_right++;
        }
    } 
    cout << "Percentage correctly classified: " <<
        100*count_right / (float) x_length << endl;
    free(labeled);
    return 0;
}

void get_fields(char* line, data_t* data) {
    int count = 0; 
    const char* tok;
    for (tok = strtok(line, ",");
            tok && *tok;
            tok = strtok(NULL, ",\n"))
    {
        data[count] = (data_t) strtod(tok, NULL);
        count++;
    }
}

