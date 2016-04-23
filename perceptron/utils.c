#include "utils.h"

#include <stdlib.h>
#include <string.h>

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

void assign_labels(data_t* x, int x_length, int x_dim, int test_case, char* y) {
    int i, j;
    for(i=0; i < x_length; ++i){ 
        switch(test_case) {
            case 1:
                y[i] = (0.2*(x[i*x_dim + 1] - 0.5)) +
                    (.6-x[i*x_dim + 2]) > 0 ? 1 : -1;
                break;
            case 2:
                y[i] = (x[i*x_dim + 1]-.5)*(x[i*x_dim + 1]-.5) +
                    (x[i*x_dim + 2]-.5)*(x[i*x_dim + 2]-.5) > 0.09 ? 1 : -1;
                break;
            case 3:
                y[i] = 4*(x[i*x_dim + 1]-.5)*(x[i*x_dim + 1]-.5) +
                    (.2-x[i*x_dim + 2]) > 0 ? 1 : -1;
                break;
            default:
                y[i] = 0;
        }
    }
}
