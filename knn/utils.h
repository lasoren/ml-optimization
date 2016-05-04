
#include <time.h>
typedef double data_t;

// Get fields from a line and fill an array with the values.
void get_fields(char* line, data_t* data, int max_width);

const char* getfield(char* line, int num);

struct timespec diff(struct timespec start, struct timespec end);
