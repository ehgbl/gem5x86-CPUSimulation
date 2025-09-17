/*
 * Sample Workload for Gem5 x86 Simulation
 * ========================================
 * 
 * This C program demonstrates various memory access patterns
 * and computational workloads to test CPU and memory performance.
 */










// Global arrays for memory access patterns
int *array_a;
int *array_b;
int *array_c;

void initialize_arrays() {
    printf("Initializing arrays...\n");
    array_a = malloc(ARRAY_SIZE * sizeof(int));
    array_b = malloc(ARRAY_SIZE * sizeof(int));
    array_c = malloc(ARRAY_SIZE * sizeof(int));
    
    // Initialize with random data
    srand(42);
    for (int i = 0; i < ARRAY_SIZE; i++) {
        array_a[i] = rand() % 1000;
        array_b[i] = rand() % 1000;
        array_c[i] = 0;
    }
}

void sequential_access() {
    printf("Performing sequential memory access...\n");
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < ARRAY_SIZE; i++) {
            array_c[i] = array_a[i] + array_b[i];
        }
    }
}

void random_access() {
    printf("Performing random memory access...\n");
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < ARRAY_SIZE; i++) {
            int index = rand() % ARRAY_SIZE;
            array_c[index] = array_a[index] * array_b[index];
        }
    }
}

void cache_friendly_loop() {
    printf("Performing cache-friendly operations...\n");
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < ARRAY_SIZE; i += 64) {  // Cache line size
            for (int j = 0; j < 64 && (i + j) < ARRAY_SIZE; j++) {
                array_c[i + j] = array_a[i + j] + array_b[i + j];
            }
        }
    }
}

void mathematical_operations() {
    printf("Performing mathematical operations...\n");
    double sum = 0.0;
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < ARRAY_SIZE; i++) {
            sum += sin(array_a[i]) + cos(array_b[i]);
        }
    }
    printf("Mathematical sum: %f\n", sum);
}

void memory_intensive_workload() {
    printf("Performing memory-intensive workload...\n");
    int **matrix = malloc(1000 * sizeof(int*));
    for (int i = 0; i < 1000; i++) {
        matrix[i] = malloc(1000 * sizeof(int));
    }
    
    // Matrix multiplication
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < 1000; j++) {
            matrix[i][j] = 0;
            for (int k = 0; k < 1000; k++) {
                matrix[i][j] += array_a[i % ARRAY_SIZE] * array_b[j % ARRAY_SIZE];
            }
        }
    }
    
    // Cleanup
    for (int i = 0; i < 1000; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void branch_intensive_workload() {
    printf("Performing branch-intensive workload...\n");
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < ARRAY_SIZE; i++) {
            if (array_a[i] > array_b[i]) {
                array_c[i] = array_a[i] - array_b[i];
            } else if (array_a[i] < array_b[i]) {
                array_c[i] = array_b[i] - array_a[i];
            } else {
                array_c[i] = array_a[i] * array_b[i];
            }
        }
    }
}

int main() {
    printf("Gem5 x86 Simulation Workload\n");
    printf("============================\n");
    
    clock_t start_time = clock();
    
    // Initialize data structures
    initialize_arrays();
    
    // Run different workload patterns
    sequential_access();
    random_access();
    cache_friendly_loop();
    mathematical_operations();
    memory_intensive_workload();
    branch_intensive_workload();
    
    // Calculate execution time
    clock_t end_time = clock();
    double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    
    printf("\nWorkload completed!\n");
    printf("Execution time: %.2f seconds\n", execution_time);
    printf("Total operations: %d\n", ARRAY_SIZE * ITERATIONS * 6);
    
    // Cleanup
    free(array_a);
    free(array_b);
    free(array_c);
    
    return 0;
}
