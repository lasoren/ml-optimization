
typedef double data_t;

// Get fields from a line and fill an array with the values.
void get_fields(char* line, data_t* data, int max_width);

const char* getfield(char* line, int num);

void assign_labels(data_t* x, int x_length, int x_dim, int test_case, char* y);
#if TIMING
struct timespec diff(struct timespec start, struct timespec end);
#endif
