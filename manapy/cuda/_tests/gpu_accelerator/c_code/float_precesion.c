#include <stdio.h>

void print_float_bits(float f) {
    unsigned int* bits_ptr = (unsigned int*)&f; // Treat the address of f as an unsigned int pointer
    unsigned int bits = *bits_ptr; // Read the bits representation of f

    // Print the bits in binary format
    for (int i = sizeof(float) * 8 - 1; i >= 0; i--) {
        printf("%u", (bits >> i) & 1);
        if (i == 23) printf(" "); // Add a space every 4 bits for readability
        if (i == 31) printf(" "); // Add a space every 4 bits for readability
    }
    printf("\n");
}

int main()
{
  float a = 0.0009765625;
  
  int b = *((int *)(&a));
  int sign = (b >> 31) & 1;
  int exponent = (b >> 23) & (255);
  int fraction = (b) & ((1 << 23) - 1);
  int fraction_f = ((127 << 23) | fraction);
  float fraction_ff = *((float *)(&fraction_f));

  printf("%d\n", (int)a);
  printf("%f\n", a);
  print_float_bits(a);
  printf("%d %d %d\n", sign, exponent, fraction);
  printf("%d %d %f\n", sign, (exponent - 127), fraction_ff);
}
