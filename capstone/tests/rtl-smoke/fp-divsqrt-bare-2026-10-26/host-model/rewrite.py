import re,sys
src,dst=sys.argv[1],sys.argv[2]; t=open(src).read(); n={}
def sub(pat,rep,key):
    global t
    t,k=re.subn(pat,rep,t,flags=re.M); n[key]=k
sub(r"= fdiv (?:[a-z]+ )*double ([^,\n]+), ([^,\n]+)$", r"= call double @rtl_fdiv_d(double \1, double \2)", "fdiv double")
sub(r"= fdiv (?:[a-z]+ )*float ([^,\n]+), ([^,\n]+)$", r"= call float @rtl_fdiv_s(float \1, float \2)", "fdiv float")
sub(r"call (?:[a-z]+ )*double @(?:llvm\.sqrt\.f64|sqrt)\(double ", r"call double @rtl_fsqrt_d(double ", "sqrt double")
sub(r"call (?:[a-z]+ )*float @(?:llvm\.sqrt\.f32|sqrtf)\(float ", r"call float @rtl_fsqrt_s(float ", "sqrt float")
left=len(re.findall(r"\bfdiv\b",t))+len(re.findall(r"call[^\n]*@(?:llvm\.sqrt|sqrtf?)\(",t))
t+="\ndeclare double @rtl_fdiv_d(double, double)\ndeclare float @rtl_fdiv_s(float, float)\ndeclare double @rtl_fsqrt_d(double)\ndeclare float @rtl_fsqrt_s(float)\n"
open(dst,"w").write(t)
print(src.split('/')[-1], n, "UNREPLACED", left)
sys.exit(1 if left else 0)
