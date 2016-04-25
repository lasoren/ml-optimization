#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"
#include "device_picker.h"
#include "utilsopencl.h"

#define MAX_ITERS 	10000
#define TEST_CASE 	3
#define X_DIM           6
#define X_LENGTH        500
#define DELTA		0.05
#define START_ETA	0.1
#define ETA   		1.0
#define MAX_SOURCE_SIZE (0x100000)
#include "utilsopencl.h"

const char* getfield(char* line, int num);

/*
  To compile, use shared compute cluster with the following compile directive:
  gcc -I/usr/local/apps/cuda-4.2/include/ perceptron_opencl.c utils.c -o perceptron.o -lOpenCL -lrt
*/
  
int main(){
  // Declare OpenCL variables & objects
  cl_uint deviceIndex = 0;
  char *kernelsource;
  cl_ulong time_start, time_end;
  cl_int err;            
  cl_device_id     device;       
  cl_context       context;       
  cl_command_queue commands;     
  cl_program       program;   
  cl_kernel        calculate_weights;
  cl_kernel	   classify;    
  cl_mem d_X = NULL;
  cl_mem d_Y = NULL;
  cl_mem d_W = NULL;
  cl_mem d_misclassified = NULL;
  cl_mem d_not_classified = NULL;
  cl_mem d_sum_missed = NULL;

// Host variables
  int i, j;
  int line_counter = 0;
  char not_classified_bool = 1;
  int iters = 0;
  int sum_missed_acc = 0;
  float* X;
  float* W;
  char* Y;
  char* misclassified;
  char* not_classified;
  int* sum_missed;
  float eta = ETA;
  float start_eta = START_ETA;
  float current_eta;
  float delta = DELTA;
  float exec_times[19][2];
  int sum_missed_iters[19][2];
// Declare sizes used to create buffers and allocate space on host for host arrays
  size_t X_size = X_DIM*X_LENGTH*sizeof(float);
  size_t W_size = X_DIM*sizeof(float);
  size_t Y_size = X_LENGTH*sizeof(char);
  size_t sum_missed_size = X_LENGTH*sizeof(int);

// Allocate host arrays
  X = (float *)malloc(X_size);  // Matrix containing data
  W = (float *)malloc(W_size);  // Weight matrix
  Y = (char *)malloc(Y_size);   // Array of labels for data
  misclassified = (char *)malloc(Y_size); // Array to store which data elements are misclassified each iter
  not_classified = (char *)malloc(Y_size); // Array to store how many threads have still not classified
  sum_missed = (int *)malloc(sum_missed_size);  // Keeps track of total misclassified elements

// Initialize host arrays
for(i=0;i< X_LENGTH;i++){
	misclassified[i] = 1;
	not_classified[i] = 0;
	sum_missed[i] = 0;
}
for(i=0; i < X_DIM; i++){
	W[i] = 0;
}

// Open file containing data & initialize X matrix
FILE* stream = fopen("data.csv", "r");
    char line[1024];
    while (fgets(line, 1024, stream) && line_counter < 500)
    {
        char* tmp = strdup(line);
        int idx = line_counter*X_DIM;
        X[idx] = 1.0;
        X[idx + 1] = strtod(getfield(tmp, 1), NULL);
        tmp = strdup(line);
        X[idx + 2] = strtod(getfield(tmp, 2), NULL);
        X[idx + 3] = X[idx + 1]*X[idx + 2];
        X[idx + 4] = X[idx + 1]*X[idx + 1];
        X[idx + 5] = X[idx + 2]*X[idx + 2]; 

        free(tmp);
        line_counter++;
    }

    // Choose test case, which affects how data is labeled
    int test_case = TEST_CASE;
   // assign_labels(X, X_LENGTH, X_DIM, test_case, Y);


    for(i=0; i < X_LENGTH; ++i){ 
        switch(test_case) {
            case 1:
                Y[i] = (0.2*(X[i*X_DIM + 1] - 0.5)) +
                    (.6-X[i*X_DIM + 2]) > 0 ? 1 : -1;
                break;
            case 2:
                Y[i] = (X[i*X_DIM + 1]-.5)*(X[i*X_DIM + 1]-.5) +
                    (X[i*X_DIM + 2]-.5)*(X[i*X_DIM + 2]-.5) > 0.09 ? 1 : -1;
                break;
            case 3:
                Y[i] = 4*(X[i*X_DIM + 1]-.5)*4*(X[i*X_DIM + 1]-.5) +
                    (.2-X[i*X_DIM + 2]) > 0 ? 1 : -1;
                break;
            default:
                Y[i] = 0;
        }
    }

  // Find a device (CPU or GPU)
  cl_device_id devices[MAX_DEVICES];
  unsigned numDevices = getDeviceList(devices);

  if (deviceIndex >= numDevices)
  {
    printf("Invalid device index (try '--list')\n");
    return EXIT_FAILURE;
  }

  device = devices[deviceIndex];

  char name[MAX_INFO_STRING];
  getDeviceName(device, name);
  printf("\nUsing OpenCL device: %s\n", name);

  // Create OpenCL context
  context = clCreateContext(0, 1, &device, NULL, NULL, &err);
  checkError(err, "Creating context");

  // Create OpenCL command queue
  commands = clCreateCommandQueue(context, device, 0, &err);
  checkError(err, "Creating command queue");

  // Create memory buffers for all of the device arrays/matrices; initialize all with data in their host counterparts
  d_X = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, X_size, X, &err);
  d_W = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, W_size, W, &err);
  d_Y = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Y_size, Y, &err);
  d_misclassified = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Y_size, misclassified, &err);
  d_not_classified = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Y_size, not_classified, &err);
  d_sum_missed = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sum_missed_size, sum_missed, &err);

// Read in source file (containing two kernels)
FILE *fp;
const char fileName[] = "./calculate_weights.cl";
size_t source_size;
char *source_str;
fp = fopen(fileName, "r");
if(!fp){
	fprintf(stderr, "Failed to load .cl file");
}
source_str = (char *)malloc(MAX_SOURCE_SIZE);
source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
fclose(fp);

// Create program from the source file
  program = clCreateProgramWithSource(context, 1, (const char **) &source_str, (const size_t *) &source_size, &err);
  checkError(err, "Creating program from calculate_weights.cl");

// Build program
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if(err != CL_SUCCESS){
  	size_t len;
  	char buffer[2048];

  	printf("Error: Failed to build program executable! \n%s\n", err_code(err));
  	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
  	printf("%s\n", buffer);
  	return EXIT_FAILURE;
  }

  // Create first kernel
  calculate_weights = clCreateKernel(program, "calculate_weights", &err);
  checkError(err, "Creating kernel from calculate_weights.cl");
  classify = clCreateKernel(program, "classify", &err);
  checkError(err, "Creating kernel from classify.cl");

  // Create local variables of parameterized values
  int x_dim = X_DIM;
  int x_length = X_LENGTH;
  int max_iters = MAX_ITERS;
  int k, index;
  index = 0; 
  current_eta = start_eta;
  // Start timer (timing method is clock_gettime)
  //tick();
for(k = 0; k < 19; k++){
  iters = 0;
  sum_missed_acc=0;
  not_classified_bool =1;
  for(i=0;i< X_LENGTH;i++){
	misclassified[i] = 1;
	not_classified[i] = 0;
	sum_missed[i] = 0;
  }
  for(i=0; i < X_DIM; i++){
	W[i] = 0;
  }  
  d_X = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, X_size, X, &err);
  d_W = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, W_size, W, &err);
  d_Y = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Y_size, Y, &err);
  d_misclassified = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Y_size, misclassified, &err);
  d_not_classified = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Y_size, not_classified, &err);
  d_sum_missed = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sum_missed_size, sum_missed, &err);


  tick();
  // Continue calling kernels while their is still data misclassified and iters is less than max iters
while(not_classified_bool && iters <= max_iters){

  // Increase iters and reinitialize not_classified_bool
  not_classified_bool = 0;
  iters++;

  // Set first kernel arguments
  err = clSetKernelArg(calculate_weights, 0, sizeof(X_size), &d_X);
  checkError(err, "Setting kernel arg 0 -- d_X");
  err = clSetKernelArg(calculate_weights, 1, sizeof(Y_size), &d_Y);
  checkError(err, "Setting kernel arg 1 -- d_Y");
  err = clSetKernelArg(calculate_weights, 2, sizeof(W_size), &d_W);
  checkError(err, "Setting kernel arg 2 --- d_W");
  err = clSetKernelArg(calculate_weights, 3, sizeof(Y_size), &d_misclassified);
  checkError(err, "Setting kernel arg 3 --- d_misclassified");
  err = clSetKernelArg(calculate_weights, 4, sizeof(cl_int), &x_length);
  checkError(err, "Setting kernel arg 4 --- x_length");
  err = clSetKernelArg(calculate_weights, 5, sizeof(cl_int), &x_dim);
  checkError(err, "Setting kernel arg 5 --- x_dim");
  err = clSetKernelArg(calculate_weights, 6, sizeof(cl_float), &current_eta);
  checkError(err, "Setting kernel arg 6 --- eta");
  printf("Launching calculate weights kernel!\n");

  // Set global and local size (i.e. number of work-groups and work-items)
  size_t global[2] = {X_LENGTH,1};
  size_t local[2] = {X_LENGTH, 1};

  // Launch kernel
  err = clEnqueueNDRangeKernel(commands, calculate_weights, 2, NULL, (size_t *) &global, (size_t *) &local, 0, NULL, NULL);
  checkError(err, "Enqueueing calculate_weights kernel");
  // clFinish blocks host code from executing until kernel has finished
  err = clFinish(commands);
  checkError(err, "Waiting for calculate_weights to finish");
  // Read data back to host
  err = clEnqueueReadBuffer(commands, d_W, CL_TRUE, 0, W_size, W, 0, NULL, NULL);
  printf("Weight vector\n");
  for(i=0; i < X_DIM; i++){
   	printf("%f ", W[i]);
  }
  printf("\n");

  checkError(err, "Copying Weight matrix back to host!");
  d_W = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, W_size, W, &err);

  // Set kernel arguments for second kernel
  err = clSetKernelArg(classify, 0, sizeof(X_size), &d_X);
  checkError(err, "Setting kernel arg 0 --- d_X from classify");
  err = clSetKernelArg(classify, 1, sizeof(Y_size), &d_Y);
  checkError(err, "Setting kernel arg 1 --- d_Y from classify");
  err = clSetKernelArg(classify, 2, sizeof(W_size), &d_W);
  checkError(err, "Setting kernel arg 2 --- d_W from classify");
  err = clSetKernelArg(classify, 3, sizeof(Y_size), &d_misclassified);
  checkError(err, "Setting kernel arg 3 --- d_misclassified from classify");
  err = clSetKernelArg(classify, 4, sizeof(Y_size), &d_not_classified);
  checkError(err, "Setting kernel arg 4 --- d_not_classified from classify");
  err = clSetKernelArg(classify, 5, sizeof(sum_missed_size), &d_sum_missed);
  checkError(err, "Setting kernel arg 5 --- d_sum_missed from classify");
  err = clSetKernelArg(classify, 6, sizeof(cl_int), &x_dim);
  checkError(err, "Setting kernel arg 6 --- x_dim from classify");
  err = clSetKernelArg(classify, 7, sizeof(cl_int), &x_length);
  checkError(err, "Setting kernel arg 7 --- x_length from classify");
  printf("Launching classify kernel\n");

  // Launch second kernel
  err = clEnqueueNDRangeKernel(commands, classify, 2, NULL, (size_t *) &global, (size_t *) &local, 0, NULL, NULL);
  checkError(err, "Enqueueing classify kernel");
  err = clFinish(commands);
  checkError(err, "Waiting for classify to finish");
  // Copy data back to host
  err = clEnqueueReadBuffer(commands, d_not_classified, CL_TRUE, 0, Y_size, not_classified, 0, NULL, NULL);
  checkError(err, "Copying not_classified back to host!");
  err = clEnqueueReadBuffer(commands, d_sum_missed, CL_TRUE, 0, sum_missed_size, sum_missed, 0, NULL, NULL);
  checkError(err, "Copying sum_missed back to host!");
  err = clEnqueueReadBuffer(commands, d_misclassified, CL_TRUE, 0, Y_size, misclassified, 0, NULL, NULL);
  checkError(err, "Copying misclassified back to host!");
  d_misclassified = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Y_size, misclassified, &err);
	
  // Accumulate not_classified for all the work items to see if all data has been classified
  for(i=0;i<x_length;i++){
		not_classified_bool += not_classified[i];
	}
}
// End timing
tock();
uint64_t exec_time;
exec_time = get_execution_time();

exec_times[index][0] = current_eta;
exec_times[index][1] = (float) (exec_time);
// Check how many are still misclassified (if iters reached max iters before all were classified)
for(i=0;i<x_length;i++){
	sum_missed_acc += sum_missed[i];
}

// Print results
//printf("Execution Time = %f ns", (double) exec_time);
if(sum_missed_acc == 0){
	printf("Perfectly separated data with %d iters\n", iters);
}
else{
	printf("Finished MAX_ITERS and still %d misclassified\n", sum_missed_acc);
}
sum_missed_iters[index][0] = sum_missed_acc;
sum_missed_iters[index][1] = iters;
printf("current_eta: %f	start_eta: %f index: %d delta: %f\n", current_eta, start_eta, index, delta);
	current_eta+= delta;
	index++;
// end of big for loop
}

printf("Eta:		Exec time (ns): 	Sum Missed: 	Iters: \n");
for(k =0; k < 19; k++){
	printf("%f\t\t%f\t%d\t%d\n", exec_times[k][0], exec_times[k][1], sum_missed_iters[k][0], sum_missed_iters[k][1]);
}

// Free memory
  free(calculate_weights);
  free(classify);

  clReleaseMemObject(d_W);
  clReleaseMemObject(d_X);
  clReleaseMemObject(d_Y);
  clReleaseMemObject(d_misclassified);
  clReleaseProgram(program);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);

  return EXIT_SUCCESS;
	
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
