Принял. Ниже — **сжатый, source-backed вывод**: какая стратегия сейчас выглядит правильной, какой план логичен, и какой **минимальный следующий шаг** я бы выбрал.

- [x] Сверил предложение другой LLM с текущими примитивами в кодовой базе
- [x] Выделил главный архитектурный выбор
- [x] Определил, что в предложении уже подтверждено исходниками, а что пока нет
- [x] Сформулировал практический план без лишнего scope creep
- [x] Выбрал минимальный следующий шаг с наибольшей ценностью и наименьшим риском

## Короткий вывод

### Рекомендованная стратегия
Для **Paper I** я рекомендую **принять split host-enclave направление**, но **не** в той форме, как его описала другая LLM.

Правильная формулировка на сейчас такая:

> **Берём split host-enclave architecture как основной roadmap, но первая версия должна быть не “resumable yield/resume syscall proxy”, а более простой synchronous shared-region RPC model.**

То есть:

- **да** — isolated `EM_CAPSTONE` domain
- **да** — host-side proxy/service execution в обычном RISC-V Linux userspace
- **да** — domain-local micro-runtime / micro-libc
- **нет пока** — assumptions про general-purpose `yield -> host syscall -> exact resume at same PC`
- **нет пока** — начинать сразу с `picolibc/newlib`
- **нет пока** — проектировать ABI вокруг регистров `a0..a3` как будто userspace helper их уже видит

---

# Что подтвердилось по исходникам

## 1. Split-модель в репозитории уже фактически существует
Самое важное наблюдение: это **не теория с нуля**.

В `miniweb` уже есть рабочий паттерн:

- host userspace создаёт и мапит регионы:
  - `create_region()`
  - `map_region()`
- потом шэрит их в домен:
  - `shared_region_annotated()`
- вызывает домен:
  - `call_dom()`
- после возврата из домена делает обычную Linux-работу:
  - `open/read/write/close`
- потом снова вызывает домен

Это видно в:
- `capstone/caplifive-buildroot/package/modcapstone/userspace/miniweb_frontend.c`
- `capstone/caplifive-buildroot/package/capstone-nested-enclave/capstone_split/sdom/miniweb_backend.smode.c`

Особенно показательно:
- `miniweb_frontend.c:272-329` — создание/маппинг/shared регионов
- `miniweb_frontend.c:130-135`, `174-178`, `215-219` — host вызывает домен, возвращается, делает host work, снова вызывает
- `miniweb_backend.smode.c:394-417` — домен живёт в цикле и много раз делает `SBI_EXT_CAPSTONE_DOM_RETURN`

### Следствие
У вас уже есть **доказанный substrate** для:
- shared buffers,
- shared metadata,
- многократных host↔domain раундов.

Это уже почти тот “proxy” мир, который вам нужен.

---

## 2. Shared bounce buffer лучше строить не с `malloc()` page, а через существующий region API
Другая LLM предложила: host helper сам выделяет обычную страницу и “grant it to the domain”.

Но по текущему коду более естественный и уже поддержанный путь такой:

- `create_region(len)`
- `map_region(region_id, len)`
- `shared_region_annotated(dom_id, region_id, perm, rev)`

Это проходит через:
- userspace `libcapstone.c`
- kernel module `capstone.c`
- SBI `sbi_capstone.c`

Файлы:
- `capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c`
- `capstone/caplifive-buildroot/package/modcapstone/module/capstone.c`
- `capstone/caplifive-buildroot/components/opensbi/lib/sbi/capstone-sbi/sbi_capstone.c`

### Следствие
Для v0/v1 вам **не надо изобретать новый memory grant mechanism**.
Ваш “bounce buffer” уже почти готов как **annotated shared region**.

---

## 3. General-purpose host-visible yield/resume сейчас не подтверждён
Это ключевой момент.

В user/kernel ABI я не нашёл:
- `IOCTL_DOM_RESUME`
- `resume_dom`
- `DOM_RESUME`
- явный механизм “домен yield’ит, userspace helper читает trapframe, потом resume exactly there”

Видно только:
- `IOCTL_DOM_CALL`
- `IOCTL_DOM_SCHEDULE`
- region operations

Файл:
- `capstone/caplifive-buildroot/package/modcapstone/include/capstone.h`

И в userspace wrapper:
- `call_dom()` просто делает `ioctl(IOCTL_DOM_CALL)` и получает один `retval`
- никаких register snapshots наружу не экспортируется

Файл:
- `capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c:362-369`

### Что реально есть
В OpenSBI есть:
- `SBI_EXT_CAPSTONE_DOM_RETURN`
- `return_from_domain(retval)`
  оно пишет `retval` в `caller_buf`, затем делает `__domreturnsaves(caller_dom, ...)`

Файл:
- `capstone/caplifive-buildroot/components/opensbi/lib/sbi/capstone-sbi/sbi_capstone.c:527-531`

Это доказывает **синхронный возврат к caller domain / SBI caller path**, но **не доказывает userspace-visible resumable syscall trap ABI** в смысле SGX-like OCall.

### Следствие
Сейчас слишком смело строить план вокруг:

> domain yields -> host reads a0/a1/a2/a3 -> host resumes same execution point

Это пока **не подтверждено текущим host ABI**.

---

## 4. `CAPSTONE_IHI_THREAD_YIELD` существует, но это не тот же самый механизм, что host syscall proxy
В `capstone_int_handler.c` есть:
- `CAPSTONE_IHI_THREAD_SPAWN`
- `CAPSTONE_IHI_THREAD_YIELD`

Файл:
- `capstone/caplifive-buildroot/components/opensbi/lib/sbi/capstone_int_handler.c`

Но по коду это выглядит как **thread / interrupt-handler scheduling path**, а не как userspace-observable syscall proxy ABI.

То есть:
- это интересный механизм,
- но я бы **не ставил на него первый milestone**.

---

# Где другая LLM права, а где надо скорректировать

## С чем я согласен
1. **Split host-enclave** — да, это правильное направление для Paper I.
2. **Domain-local sysroot** — да.
3. **Micro-libc first** — да.
4. **Host-side syscall execution** — да.
5. **Не протаскивать сейчас full native Capstone Linux userspace** — да.

## Что я бы изменил
### Вместо этого:
> shared bounce buffer + yield/resume + a0/a1/a2 ABI to host

### Я бы написал так:
> annotated shared regions + synchronous multi-round RPC + memory-based request/response ABI

Это гораздо ближе к тому, что уже доказано исходниками.

---

# Рекомендованная стратегия действий

## Strategy decision
Я бы **официально выбрал** такой near-term milestone:

> **Capstone enclave/domain runtime with synchronous host-call proxy over shared regions**

А не:

> **native `capstone64-unknown-linux-gnu` hosted userspace**

И не:

> **full resumable proxy kernel with in-call host OCalls**

---

# Практический план

## Phase 0 — зафиксировать реальные ограничения
Цель: не строить архитектуру на неподтверждённом resume ABI.

Что уже знаем:
- shared regions есть
- region annotations есть
- repeated `call_dom()` cycles есть
- `DOM_RETURN` есть
- userspace-visible resumable trap ABI пока не доказан

### Вывод Phase 0
Первую версию строим как **state-machine RPC**, не как “mid-call trap/resume”.

---

## Phase 1 — HostCall ABI v0
Сделать минимальный ABI **целиком через shared memory**, а не через host-visible registers.

### Рекомендованный ABI v0
Два региона:

1. **metadata region** (`INOUT`, `SHARED`)
2. **bounce buffer region** (`INOUT`, `SHARED` на первом этапе для простоты)

### Metadata layout
Например:

```c
struct hostcall_v0 {
    uint64_t phase;      // INIT / REQ / RESP / DONE / ERROR
    uint64_t opcode;     // HC_WRITE_STDOUT = 1
    uint64_t offset;     // into bounce buffer
    uint64_t length;     // bytes
    int64_t  result;     // host return value
    int64_t  error;      // errno-like
};
```

### Flow
1. host создаёт домен
2. host создаёт 2 shared regions
3. host шарит их в домен
4. host делает `call_dom(dom_id)`
5. domain записывает request в metadata + payload в buffer
6. domain делает `DOM_RETURN(HC_PENDING)`
7. host читает metadata/buffer, выполняет Linux `write(1, ...)`
8. host пишет `result/error` обратно в metadata
9. host снова делает `call_dom(dom_id)`
10. domain видит response и завершает вызов `puts()`

### Почему это лучший v0
- не требует нового kernel ABI
- не требует `DOM_RESUME`
- не требует export trapframe
- полностью опирается на уже существующие примитивы

---

## Phase 2 — micro-libc, но только после ABI proof
Пока не брать `picolibc`/`newlib`.

Сначала:
- `strlen`
- `memcpy`
- `puts`

И только когда `puts()` реально работает через host proxy:
- смотреть на `picolibc`
- `newlib` как план B

### Почему не сразу `picolibc`
Потому что главный риск сейчас — **не libc**, а **host-call contract**.

---

## Phase 3 — расширение до `write` и простого file API
После `puts()`:

1. `write(fd, buf, len)`
2. возможно `read`
3. возможно `open/close`
4. потом уже думать о `sqlite`-friendly contract

Для `sqlite` я бы заранее ожидал, что вам нужен либо:
- subset POSIX,
- либо отдельный host-backed VFS

---

# Минимальный следующий шаг

## Мой выбор: не кодить сразу libc, а сделать **один крошечный source-backed PoC**
Самый правильный минимальный следующий шаг:

> **Сделать v0 “hosted puts” prototype как synchronous two-step RPC поверх existing shared-region API.**

Не full proxy kernel.
Не `picolibc`.
Не `sqlite`.
Не yield/resume.

### Что именно должно быть в этом шаге
#### На host side
Мини-харнесс:
- `create_dom(...)`
- `create_region(metadata)`
- `create_region(buffer)`
- `map_region(...)`
- `shared_region_annotated(...)`
- `call_dom(...)`
- если domain вернул `HC_PENDING`, host печатает `buffer[offset:offset+len]`
- пишет `result` в metadata
- `call_dom(...)` ещё раз

#### На domain side
Мини-домен без libc:
- локальная строка `"hello from domain\n"`
- `memcpy` в shared buffer
- запись `opcode/offset/len` в metadata
- `DOM_RETURN(HC_PENDING)`
- при следующем вызове читает `result`
- завершает работу

### Почему именно это
Это:
- подтверждает жизнеспособность архитектуры,
- использует уже существующие примитивы,
- убирает главный риск,
- сразу даёт очень сильный handoff/demo artifact.

---

# Если сформулировать совсем жёстко

## Что **не** надо делать следующим шагом
- не делать сейчас `capstone64-unknown-linux-gnu` hosted bring-up
- не делать сейчас `glibc`/`musl`/`picolibc` port
- не проектировать ABI вокруг host-visible `a0..a7`
- не предполагать, что `domreturn` = resumable syscall yield
- не обещать `sqlite` до proof-of-concept `puts`

## Что **надо** делать следующим шагом
- использовать existing region sharing substrate
- формализовать HostCall ABI v0 в shared memory
- сделать one-request prototype: `HC_WRITE_STDOUT`
- только после этого двигаться к micro-libc

---

# Мой итоговый verdict

## Стратегия
**Да — split host-enclave.**

## Но с поправкой
**Первая версия должна быть synchronous shared-region RPC, а не speculative yield/resume proxy kernel.**

## План
1. Зафиксировать v0 ABI
2. Сделать `puts` PoC без libc
3. Обернуть это в micro-libc
4. Потом выбирать `picolibc` / `newlib`
5. Только потом расширять к `write/open/read/...`

## Минимальный следующий шаг
**Сделать “hosted puts via shared metadata + bounce buffer + two `call_dom()` rounds” prototype.**

---

Если хочешь, следующим сообщением я могу сделать уже **совсем конкретный engineering spec для шага v0**:

1. точный layout `metadata` структуры
2. точные значения `opcode/status`
3. рекомендуемые `annotation_perm/annotation_rev` для двух регионов
4. псевдокод host harness
5. псевдокод domain-side `puts` stub
6. критерий успеха для первого эксперимента.

