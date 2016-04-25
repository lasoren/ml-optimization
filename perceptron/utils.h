
typedef double data_t;

const char* getfield(char* line, int num);

void assign_labels(data_t* x, int x_length, int x_dim, int test_case, char* y);

struct timespec diff(struct timespec start, struct timespec end);
