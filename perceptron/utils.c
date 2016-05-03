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

void get_fields(char* line, data_t* data, int max_width) {
    int count = 0; 
    const char* tok;
    for (tok = strtok(line, ",");
            tok && *tok;
            tok = strtok(NULL, ",\n"))
    {
        data[count] = (data_t) strtod(tok, NULL);
        count++;
        if (count == max_width) {
            break;
        }
    }
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
                    (x[i*x_dim + 2]-.5)*(x[i*x_dim + 2]-.5)
                    > 0.09 ? 1 : -1;
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

#if TIMING
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
#endif
