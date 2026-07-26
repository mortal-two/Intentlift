#include <inttypes.h>
#include <riscv_vector.h>
#include <stdio.h>

int main(void) {
  size_t e8m1 = __riscv_vsetvlmax_e8m1();
  size_t e16m1 = __riscv_vsetvlmax_e16m1();
  size_t e32m1 = __riscv_vsetvlmax_e32m1();
  size_t e64m1 = __riscv_vsetvlmax_e64m1();
  printf("vlmax_e8m1=%zu\n", e8m1);
  printf("vlmax_e16m1=%zu\n", e16m1);
  printf("vlmax_e32m1=%zu\n", e32m1);
  printf("vlmax_e64m1=%zu\n", e64m1);
  printf("vlen_bits=%zu\n", e8m1 * 8);
  return 0;
}
