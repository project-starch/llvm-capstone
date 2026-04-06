.section .text.entry
    .global _start
_start:
    # 1. При старте домена в ra (x1) лежит Exit Capability.
    # Сохраняем ее в регистр s1 (x9), чтобы не потерять при вызове main.
    movc s1, ra

    # 2. Инициализируем s0 как null-capability (x0).
    # Ваш LLVM генерирует 'stc s0, 16(sp)' в прологе main.
    # STC требует, чтобы сохраняемый регистр был capability.
    movc s0, zero

    # 3. Вызываем вашу C-функцию main!
    # Ваш ассемблер (clang) поддерживает стандартные макросы, 
    # jal сама использует pc (которая является capability)
    jal ra, main

    # 4. Выходим из домена обратно в Linux!
    # s1 - наша спасенная Exit Capability
    # a0 - результат функции main (число 42)
    capexit s1, a0
