# fb46cda19602 — When we're in a for(i=0,[...]) loop, don't reuse (and thus reset) 'i' in another for(i=0,[...]) loop.  This fixes the capinfos double-free crashes that the fuzz bot has been experiencing.

## The defect

`wtap_close()` declared a single `gint i;` and used it for both the interface loop and the nested interface-statistics loop. Finishing the inner loop therefore rewinds the outer one, and the description strings freed on the first pass — none of which is set to NULL — are freed again.

## Upstream defect

- **Fix:** `fb46cda19602`, *"When we're in a for(i=0,[...]) loop, don't reuse (and thus reset) 'i' in another for(i=0,[...]) loop.  This fixes the capinfos double-free crashes that the fuzz bot has been experiencing."*, `wiretap/wtap.c`.
- **CVE:** none assigned.
- **Live at our v4.6.8 pin: NO.** The construct was replaced wholesale at the pin.

## The vulnerable code, quoted from the fix's parent

```c
	gint i;
	...
	for(i = 0; i < (gint)wth->number_of_interfaces; i++) {
	...
		for(i = 0; i < (gint)wtapng_if_descr->num_stat_entries; i++) {
```

## The fix

```c
	gint i, j;
	...
		for(j = 0; j < (gint)wtapng_if_descr->num_stat_entries; j++) {
```

## What is real here, and what is reduced

**Real:** which allocation's lifetime ends, the call that ends it, the pointer left holding the
freed address, and the access that follows. The object is a plain allocation because upstream's is
— this code is outside `epan`'s wmem scopes.

**Reduced:** no capture file and no interface descriptions. Two loops share one index as upstream's did, one allocation stands for a description string, and the repeated pass's release is reduced to a read through the stale pointer.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — on the buggy arm the
stale pointer reaches storage that now belongs to a different live object, and under the fix it
does not. The arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is explicitly conditional on revocation sweep timing. Nor upstream reachability of
the specific sequence chosen here.
