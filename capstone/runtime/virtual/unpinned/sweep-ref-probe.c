/* Test-only observer. It takes its own temporary page reference, reports the
 * refcount including that reference, and releases it before returning. The
 * native client compares the same exclusive anonymous page before and after
 * a stopped namespace sweep. This is not a migration/concurrency stress test. */
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/pgtable.h>
#include <linux/miscdevice.h>
#include <linux/uaccess.h>
#include "sweep-ref-probe.h"

static long sample(struct file *file, unsigned int cmd, unsigned long arg)
{
    struct sweep_ref_sample value;
    struct mm_struct *mm = current->mm;
    struct page *page;
    pgd_t *pgd;
    p4d_t *p4d;
    pud_t *pud;
    pmd_t *pmd;
    pte_t *pte, entry;
    spinlock_t *lock;
    int error = -EFAULT;
    if (cmd != SWEEP_REF_SAMPLE) return -ENOTTY;
    if (copy_from_user(&value, (void __user *)arg, sizeof(value))) return -EFAULT;
    mmap_read_lock(mm);
    /* A single-page observer: no GUP touch/activation and no tag-summary
     * query. The client cross-checks this PFN against the namespace probe,
     * whose independent GUP lookup supplies the seed's physical address. */
    pgd = pgd_offset(mm, value.address);
    if (pgd_none(*pgd) || pgd_bad(*pgd)) goto out;
    p4d = p4d_offset(pgd, value.address);
    if (p4d_none(*p4d) || p4d_bad(*p4d)) goto out;
    pud = pud_offset(p4d, value.address);
    if (pud_none(*pud) || pud_leaf(*pud) || pud_bad(*pud)) goto out;
    pmd = pmd_offset(pud, value.address);
    if (pmd_none(*pmd) || pmd_leaf(*pmd) || pmd_bad(*pmd)) goto out;
    pte = pte_offset_map_lock(mm, pmd, value.address, &lock);
    if (!pte) goto out;
    entry = READ_ONCE(*pte);
    if (!pte_present(entry) || pte_special(entry) || pte_devmap(entry) ||
        !pfn_valid(pte_pfn(entry))) goto unlock;
    page = pfn_to_page(pte_pfn(entry));
    if (PageCompound(page) || !PageAnon(page)) goto unlock;
    get_page(page);
    value.references = page_ref_count(page);
    value.mappings = page_mapcount(page);
    value.pfn = page_to_pfn(page);
    put_page(page);
    error = 0;
unlock:
    pte_unmap_unlock(pte, lock);
out:
    mmap_read_unlock(mm);
    if (error) return error;
    return copy_to_user((void __user *)arg, &value, sizeof(value)) ? -EFAULT : 0;
}
static const struct file_operations ops = {
    .owner = THIS_MODULE, .unlocked_ioctl = sample,
};
static struct miscdevice device = {
    .minor = MISC_DYNAMIC_MINOR, .name = "sweep-ref-probe", .fops = &ops, .mode = 0600,
};
static int __init init(void) { return misc_register(&device); }
static void __exit fini(void) { misc_deregister(&device); }
module_init(init);
module_exit(fini);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Test-only unpinned sweep reference observer");
