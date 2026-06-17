int initialise_benchmark()
{
  cast128_set_key(&cast128_ctx, CAST128_KEY_SIZE, key);
  return 0;
}

int
benchmark(void)
{
  cast128_encrypt(&cast128_ctx, CAST128_KEY_SIZE, result, data);
  cast128_decrypt(&cast128_ctx, CAST128_KEY_SIZE, result, result);
  return 0;
}

int verify_benchmark(int unused)
{
  (void)unused;
  for (int i = 0; i < 16; i++)
    if (result[i] != (uint8_t)i)
      return 0;
  return 1;
}
