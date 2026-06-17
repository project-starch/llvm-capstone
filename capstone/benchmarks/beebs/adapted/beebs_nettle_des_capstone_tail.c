int verify_benchmark(int unused)
{
  (void)unused;
  static uint8_t expected[16] =
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  for (int i = 0; i < 16; i++)
    if (result[i] != expected[i])
      return 0;
  return 1;
}

