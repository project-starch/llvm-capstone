"""Sum reviewed half-open instruction ranges in saved heap annotations."""
from pathlib import Path
import json,re,sys
root=Path(sys.argv[1]); result={}
# Addresses below identify the base image's reviewed stage boundaries. Offsets
# from each function's first instruction also apply to an otherwise unchanged
# function relocated in another image. enframe is supplied per image below.
ranges={
'__capstone_malloc_slot_take': [
 ('entry and clean flags',0x213ec,0x21424),
 ('reuse scrub handoff and flag clear',0x21424,0x2144c),
 ('return parent',0x2144c,0x21468)],
'__capstone_malloc_slot_key': [
 ('last result, nominal size and key check',0x1e78c,0x1e7bc),
 ('hash constant and old-key check',0x1e7bc,0x1e7e8),
 ('unlink old key',0x1e7e8,0x1e84c),
 ('index insert',0x1e84c,0x1e884),
 ('return and failure',0x1e884,0x1e88c)],
'__capstone_malloc_slot_release': [
 ('key check and hash bucket',0x21470,0x214c8),
 ('find record in chain',0x214c8,0x214f4),
 ('unlink and clear key',0x214f4,0x21504),
 ('clear kind and return',0x21504,0x21510)],
'free': [
 ('entry and errno save',0x22170,0x221b8),
 ('heap lock',0x221b8,0x22200),
 ('checked handoff',0x22200,0x22214),
 ('kind check, CREVOKE and child clear',0x22214,0x22244),
 ('recover slot cursor and free_slot handoff',0x22244,0x22264),
 ('unlock and errno restore',0x22264,0x22280),
 ('return',0x22280,0x222ac)],
'__capstone_mallocng_malloc': [
 ('entry, request checks and large path',0x2b54d8,0x2b5870),
 ('size class',0x2b5870,0x2b594c),
 ('backend lock check',0x2b594c,0x2b5990),
 ('active group selection',0x2b5990,0x2b5b44),
 ('available slot and alloc_slot handoff',0x2b5b44,0x2b5bcc),
 ('failure path',0x2b5bcc,0x2b5c88),
 ('unlock, enframe handoff and return',0x2b5c88,0x2b5cf4)],
'__capstone_mallocng_free_slot': [
 ('entry and stride',0x2b4538,0x2b467c),
 ('slot index and slot_start handoff',0x2b467c,0x2b4694),
 ('prefix and footer validation',0x2b4694,0x2b471c),
 ('prefix invalidation and slot_release handoff',0x2b471c,0x2b4774),
 ('free mask and group-release path',0x2b4774,0x2b48e4),
 ('return',0x2b48e4,0x2b490c)],
'publish': [
 ('result record binding and clean flag',0x22c4c,0x22d3c),
 ('replace old child if present',0x22d3c,0x22d4c),
 ('record fields and node budget',0x22d4c,0x22da0),
 ('derive checks, CDERIVE and return',0x22da0,0x22ddc)],
'checked': [
 ('entry',0x222ac,0x222bc),
 ('caller capability validation load',0x222bc,0x222c0),
 ('address lookup and public record check',0x222c0,0x22340),
 ('return and failure path',0x22340,0x2235c)]}
if root.name.endswith('rv64'):
    ranges={
      '__libc_malloc_impl': [
       ('entry, request checks and large path',0x2396c4,0x2397a0),
       ('size class',0x2397a0,0x2398ac),
       ('backend lock check',0x2398ac,0x2398c4),
       ('active group selection',0x2398c4,0x23997c),
       ('available slot and alloc_slot handoff',0x23997c,0x2399dc),
       ('unlock, enframe handoff and return',0x2399dc,0x239a28)],
      '__libc_free': [
       ('entry and get_meta handoff',0x238d48,0x238d6c),
       ('stride and prefix',0x238d6c,0x238db4),
       ('slot end and footer validation',0x238db4,0x238e24),
       ('prefix invalidation and masks',0x238e24,0x238e48),
       ('free mask and group-release path',0x238e48,0x238ef8),
       ('return',0x238ef8,0x238f14)]}
if len(sys.argv)>2 and sys.argv[2] != '-':
    ranges['enframe']=json.loads(Path(sys.argv[2]).read_text())
for name,parts in ranges.items():
    text=(root/(name+'.txt')).read_text()
    total,entries=map(int,re.search(r': (\d+) attempts, (\d+) entries',text).groups())
    rows=[(int(pc,16),int(n)) for n,pc in re.findall(r'^\s*(\d+)\s+([0-9a-f]+):',text,re.M)]
    assert sum(n for _,n in rows)==total
    slide=rows[0][0]-parts[0][1]
    values=[]
    for label,lo,hi in parts:
        count=sum(n for pc,n in rows if lo+slide<=pc<hi+slide)
        values.append(dict(stage=label,attempts=count,per_entry=count/entries,
                           offset_start=lo-parts[0][1],offset_end=hi-parts[0][1]))
    assert sum(v['attempts'] for v in values)==total,(name,total,values)
    result[name]=dict(attempts=total,entries=entries,stages=values)
    print(name,total,entries)
    for v in values: print(' ',v['stage'],v['attempts'],f"{v['per_entry']:.4f}")
if len(sys.argv)>3: Path(sys.argv[3]).write_text(json.dumps(result,indent=2)+'\n')
