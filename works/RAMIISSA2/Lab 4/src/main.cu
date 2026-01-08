#include <cuda_runtime.h>
#include <vector>
#include <iostream>


void run_all();
void warmup_thrust();
void warmup_radix();

int main() {
    // CUDA context warm-up
    std::cout << "\nCUDA context warm-up ...";
    cudaFree(0);

    // Thrust warm-up
    std::cout << "\nThrust warm-up ...";
    warmup_thrust();

    // Radix warm-up
    std::cout << "\nRadix warm-up ...";
    warmup_radix();

    std::cout << std::endl << "---------------------------------------------------------------------";;

    run_all();
    return 0;
}
