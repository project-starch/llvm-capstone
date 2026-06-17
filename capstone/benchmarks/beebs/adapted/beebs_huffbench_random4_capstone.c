static size_t random4()
{
  seed = seed * 1103515245L + 12345L;
  return (size_t)(seed & 31);
}

