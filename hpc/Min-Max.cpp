#include <iostream>
#include <vector>
#include <omp.h>
#include <limits>

using namespace std;

int main() {
    int n = 100000;
    vector<int> arr(n);

    // Initialize array
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
    }

    int sum = 0;
    int min_val = INT_MAX;
    int max_val = INT_MIN;

    double start = omp_get_wtime();

    // 🔹 Parallel Reduction
    #pragma omp parallel for reduction(+:sum) reduction(min:min_val) reduction(max:max_val)
    for (int i = 0; i < n; i++) {
        sum += arr[i];

        if (arr[i] < min_val)
            min_val = arr[i];

        if (arr[i] > max_val)
            max_val = arr[i];
    }

    double end = omp_get_wtime();

    double avg = (double)sum / n;

    cout << "Sum = " << sum << endl;
    cout << "Minimum = " << min_val << endl;
    cout << "Maximum = " << max_val << endl;
    cout << "Average = " << avg << endl;
    cout << "Execution Time = " << end - start << " seconds" << endl;

    return 0;
}

//How to run -
// g++ HPC3.cpp -o HPC3 -fopenmp
//.\HPC3.exe
