#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <csignal>

// For system calls and timing
#include <unistd.h>

// For shared memory (IPC)
#include <sys/ipc.h>
#include <sys/shm.h>

// For NVIDIA Management Library (NVML)
#include <nvml.h>

// --- Sliding Window Class Implementation ---
// Manages a history of N power measurements using an efficient circular buffer.
class SlidingWindow {
private:
    std::vector<unsigned int> buffer; // Stores the power measurements
    size_t head;                      // Points to the next insertion spot
    size_t count;                     // Current number of elements in the buffer
    const size_t capacity;            // Total capacity of the window (N)
    const size_t half_capacity;       // Size of each sub-window (N/2)

public:
    // Constructor
    SlidingWindow(size_t n) : capacity(n), half_capacity(n / 2), head(0), count(0) {
        if (n == 0 || n % 2 != 0) {
            throw std::invalid_argument("Window size N must be a positive even number.");
        }
        buffer.resize(n, 0); // Pre-allocate and initialize to 0
    }

    // Adds a new power measurement, overwriting the oldest if full.
    void add(unsigned int value) {
        buffer[head] = value;
        head = (head + 1) % capacity;
        if (count < capacity) {
            count++;
        }
    }

    // Returns true if the window has collected N measurements.
    bool is_full() const {
        return count == capacity;
    }

    // Calculates the sum of the 'previous' (older) half of the window.
    long long get_prev_sum() const {
        if (!is_full()) return 0;
        long long sum = 0;
        // The oldest data starts at the current 'head' position.
        size_t start_index = head;
        for (size_t i = 0; i < half_capacity; ++i) {
            sum += buffer[(start_index + i) % capacity];
        }
        return sum;
    }

    // Calculates the sum of the 'current' (newer) half of the window.
    long long get_current_sum() const {
        if (!is_full()) return 0;
        long long sum = 0;
        // The newer data starts halfway through the circular buffer.
        size_t start_index = (head + half_capacity) % capacity;
        for (size_t i = 0; i < half_capacity; ++i) {
            sum += buffer[(start_index + i) % capacity];
        }
        return sum;
    }
};


// Global pointer to the shared memory segment
volatile int* shared_flag = nullptr;
int shmid = -1; // Shared memory ID

// Signal handler to ensure clean shutdown
void signalHandler(int signum) {
    std::cout << "\nController shutting down..." << std::endl;
    if (shared_flag) {
        shmdt(const_cast<int*>(shared_flag));
    }
    if (shmid != -1) {
        shmctl(shmid, IPC_RMID, NULL);
    }
    nvmlShutdown();
    exit(signum);
}

int main(int argc, char* argv[]) {
    // --- Configuration ---
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " --gpu-id <id>" << std::endl;
        return 1;
    }
    int gpu_id = std::stoi(argv[2]);

    // --- NEW Sliding Window Configuration ---
    // Total number of historical measurements to keep.
    const size_t WINDOW_SIZE_N = 20; // e.g., 20 measurements * 50ms = 1 second of history

    // Activate if the sum of power in the newest 10 readings is less than the
    // sum of the oldest 10 readings by this amount. This detects a rapid drop.
    const int POWER_DROP_THRESHOLD_WATTS = 500; // A 500W aggregate drop

    const unsigned int POLLING_INTERVAL_US = 50000; // 50ms polling

    // --- Setup ---
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    SlidingWindow power_window(WINDOW_SIZE_N);

    // --- Setup Shared Memory ---
    key_t key = ftok("firefly_ipc_key", gpu_id);
    if (key == -1) { perror("ftok"); return 1; }
    shmid = shmget(key, sizeof(int), IPC_CREAT | 0666);
    if (shmid == -1) { perror("shmget"); return 1; }
    shared_flag = (int*)shmat(shmid, (void*)0, 0);
    if (shared_flag == (int*)(-1)) { perror("shmat"); return 1; }
    *shared_flag = 0;
    std::cout << "Controller for GPU " << gpu_id << " started with sliding window." << std::endl;

    // --- Setup NVML ---
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) { std::cerr << "NVML Init Error: " << nvmlErrorString(result) << std::endl; return 1; }
    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS) { std::cerr << "NVML Handle Error: " << nvmlErrorString(result) << std::endl; nvmlShutdown(); return 1; }
    
    // --- Main Control Loop ---
    std::cout << "Monitoring power on GPU " << gpu_id << ". Press Ctrl+C to exit." << std::endl;
    while (true) {
        unsigned int power_milliwatts;
        result = nvmlDeviceGetPowerUsage(device, &power_milliwatts);

        if (result == NVML_SUCCESS) {
            unsigned int power_watts = power_milliwatts / 1000;
            power_window.add(power_watts);

            // Only make a decision after the window is full of data.
            if (power_window.is_full()) {
                long long prev_sum = power_window.get_prev_sum();
                long long current_sum = power_window.get_current_sum();
                
                // A large negative difference indicates a rapid power drop.
                // The threshold is negative because we're checking for a drop.
                if ((current_sum - prev_sum) < -POWER_DROP_THRESHOLD_WATTS) {
                    *shared_flag = 1; // Signal RUN
                } else {
                    *shared_flag = 0; // Signal STOP
                }
            } else {
                // Before window is full, remain in STOP state for safety.
                *shared_flag = 0;
            }
        } else {
            std::cerr << "Failed to get power for GPU " << gpu_id << ": " << nvmlErrorString(result) << std::endl;
            *shared_flag = 0; 
        }

        usleep(POLLING_INTERVAL_US);
    }

    signalHandler(0);
    return 0;
}

