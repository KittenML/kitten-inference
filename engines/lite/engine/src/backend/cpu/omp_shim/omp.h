// omp.h for the kitten OpenMP shim. Used where no OpenMP runtime exists (iOS with Apple clang): compile
// with `-Xclang -fopenmp -I<this dir>` and link kmp_shim.cpp. Declares only the OpenMP API the engine uses (plus a few
// common queries). The compiler-emitted __kmpc_* entry points are implemented in kmp_shim.cpp.
#ifndef KITTEN_OMP_SHIM_H
#define KITTEN_OMP_SHIM_H
#ifdef __cplusplus
extern "C" {
#endif
void   omp_set_num_threads(int n);
int    omp_get_num_threads(void);
int    omp_get_max_threads(void);
int    omp_get_thread_num(void);
int    omp_get_num_procs(void);
int    omp_in_parallel(void);
void   omp_set_dynamic(int on);
int    omp_get_dynamic(void);
int    omp_get_level(void);
int    omp_get_active_level(void);
void   omp_set_max_active_levels(int n);
int    omp_get_max_active_levels(void);
int    omp_get_thread_limit(void);
double omp_get_wtime(void);
#ifdef __cplusplus
}
#endif
#endif
