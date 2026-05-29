#include "coremark.h"

typedef ee_s32 (*list_cmp)(list_data *a, list_data *b, core_results *res);

extern ee_s32 cmp_idx(list_data *a, list_data *b, core_results *res);
extern list_head *core_list_mergesort(list_head *list, list_cmp cmp,
                                      core_results *res);
extern list_head *core_list_insert_new(list_head *insert_point, list_data *info,
                                       list_head **memblock,
                                       list_data **datablock,
                                       list_head *memblock_end,
                                       list_data *datablock_end);

list_head *core_list_init(ee_u32 blksize, list_head *memblock, ee_s16 seed) {
  /*
   * Upstream CoreMark hard-codes 16 bytes of pointer storage in per_item to
   * cover 64-bit pointer targets. Capstone PureCap pointers are wider, so the
   * original formula overestimates the number of list cells that fit in the
   * block and walks the datablock cursor out of bounds during initialization.
   */
  ee_u32 per_item = (ee_u32)(sizeof(list_head) + sizeof(list_data));
  ee_u32 size = (blksize / per_item) - 2;
  list_head *memblock_end = memblock + size;
  list_data *datablock = (list_data *)(memblock_end);
  list_data *datablock_end = datablock + size;
  ee_u32 i;
  list_head *finder, *list = memblock;
  list_data info;

  list->next = NULL;
  list->info = datablock;
  list->info->idx = 0x0000;
  list->info->data16 = (ee_s16)0x8080;
  memblock++;
  datablock++;
  info.idx = 0x7fff;
  info.data16 = (ee_s16)0xffff;
  core_list_insert_new(list, &info, &memblock, &datablock, memblock_end,
                       datablock_end);

  for (i = 0; i < size; i++) {
    ee_u16 datpat = ((ee_u16)(seed ^ i) & 0xf);
    ee_u16 dat = (datpat << 3) | (i & 0x7);
    info.data16 = (dat << 8) | dat;
    core_list_insert_new(list, &info, &memblock, &datablock, memblock_end,
                         datablock_end);
  }

  finder = list->next;
  i = 1;
  while (finder->next != NULL) {
    if (i < size / 5)
      finder->info->idx = i++;
    else {
      ee_u16 pat = (ee_u16)(i++ ^ seed);
      finder->info->idx = 0x3fff & (((i & 0x07) << 8) | pat);
    }
    finder = finder->next;
  }

  list = core_list_mergesort(list, cmp_idx, NULL);
  return list;
}


