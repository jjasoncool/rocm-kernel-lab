// hip-operator-lab / vector_add.cpp
// Day 1: HIP Vector Addition — 你的第一個 GPU kernel
//
// 編譯: hipcc vector_add.cpp -o vector_add
// 執行: ./vector_add
// Profiling: rocprof --stats ./vector_add

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ---------------------------------------------------------------------------
// HIP 錯誤檢查巨集 — 每個 HIP API call 都要包這個，養成好習慣
// ---------------------------------------------------------------------------
#define HIP_CHECK(call)                                                      \
    do {                                                                      \
        hipError_t err = (call);                                              \
        if (err != hipSuccess) {                                              \
            fprintf(stderr, "HIP error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, hipGetErrorString(err));              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// GPU Kernel: 每個 thread 負責一個元素的加法
// ---------------------------------------------------------------------------
// 思考: 為什麼這裡用 __global__？跟 CPU 的 function 有什麼不同？
// 答案: __global__ 表示這個函式在 GPU 上執行，由 CPU 端呼叫（launch）
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // 計算這個 thread 負責的 index
    // blockDim.x = 每個 block 裡有幾個 thread
    // blockIdx.x = 這是第幾個 block
    // threadIdx.x = 在這個 block 裡是第幾個 thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 邊界檢查: n 不一定是 blockDim 的倍數，多餘的 thread 不做事
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ---------------------------------------------------------------------------
// CPU 版本: 用來驗證 GPU 結果是否正確
// ---------------------------------------------------------------------------
void vector_add_cpu(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// ---------------------------------------------------------------------------
// 驗證: 比較 GPU 和 CPU 的結果
// ---------------------------------------------------------------------------
bool verify(const float* gpu_result, const float* cpu_result, int n,
            float tolerance = 1e-5f) {
    for (int i = 0; i < n; i++) {
        if (fabs(gpu_result[i] - cpu_result[i]) > tolerance) {
            fprintf(stderr, "Mismatch at index %d: GPU=%.6f, CPU=%.6f\n",
                    i, gpu_result[i], cpu_result[i]);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    // --- 參數設定 ---
    const int N = 1 << 20;  // 1M 個元素（試試改成 1<<24 看效能變化）
    const int bytes = N * sizeof(float);

    // --- 印出 GPU 資訊 ---
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("=== hip-operator-lab: vector_add ===\n");
    printf("GPU: %s\n", prop.name);
    printf("Compute units: %d\n", prop.multiProcessorCount);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Vector size: %d elements (%.1f MB)\n\n", N, (float)bytes / 1e6);

    // --- Step 1: 在 CPU (host) 上分配記憶體並初始化 ---
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c_gpu = (float*)malloc(bytes);  // 存 GPU 算完拷回來的結果
    float* h_c_cpu = (float*)malloc(bytes);  // 存 CPU 的結果，用來驗證

    for (int i = 0; i < N; i++) {
        h_a[i] = (float)(rand()) / RAND_MAX;
        h_b[i] = (float)(rand()) / RAND_MAX;
    }

    // --- Step 2: 在 GPU (device) 上分配記憶體 ---
    // 思考: 為什麼不能直接把 h_a 傳給 kernel？
    // 答案: CPU 和 GPU 有各自獨立的記憶體空間，資料要明確複製過去
    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));

    // --- Step 3: 把資料從 CPU 複製到 GPU ---
    HIP_CHECK(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice));

    // --- Step 4: 設定 launch 參數並執行 kernel ---
    const int BLOCK_SIZE = 256;  // 每個 block 256 個 thread（常見預設值）
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 計算需要多少 block

    // 用 hipEvent 計時 — 這是 GPU 計時的標準做法
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));
    // launch kernel: <<<grid 大小, block 大小>>>
    vector_add<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float gpu_ms = 0;
    HIP_CHECK(hipEventElapsedTime(&gpu_ms, start, stop));

    // --- Step 5: 把結果從 GPU 複製回 CPU ---
    HIP_CHECK(hipMemcpy(h_c_gpu, d_c, bytes, hipMemcpyDeviceToHost));

    // --- Step 6: CPU 計算 + 驗證 ---
    vector_add_cpu(h_a, h_b, h_c_cpu, N);

    if (verify(h_c_gpu, h_c_cpu, N)) {
        printf("[PASS] GPU result matches CPU reference\n");
    } else {
        printf("[FAIL] Results mismatch!\n");
    }

    // --- 效能報告 ---
    float bandwidth_gb = 3.0f * bytes / (gpu_ms * 1e-3f) / 1e9f;
    printf("\nPerformance:\n");
    printf("  Kernel time:       %.3f ms\n", gpu_ms);
    printf("  Effective BW:      %.1f GB/s\n", bandwidth_gb);
    printf("  Elements/second:   %.2f G\n", N / (gpu_ms * 1e-3f) / 1e9f);

    // --- 清理 ---
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    free(h_a);
    free(h_b);
    free(h_c_gpu);
    free(h_c_cpu);

    printf("\n--- Day 1 complete! ---\n");
    printf("Next: 試著改 BLOCK_SIZE (64, 128, 512, 1024) 看效能怎麼變化\n");
    printf("Next: 把 N 改成 1<<24，觀察 bandwidth 數字的變化\n");

    return 0;
}
