import sympy as sp, time
u=sp.symbols('u', positive=True)
s=sp.symbols('s')
a,z=sp.symbols('a z', positive=True)
d=sp.Integer(1)
h=a+z+1
m=a+z-1
c=sp.sqrt(2*(a+1)*(a+z-1))
gamma=-2*a/c
lamU=-2/c
lamY=sp.sqrt(2*(a+z-1)/(a+1))
V0=(a+z)*(a+z+1)
Nf,Sf,Pf=sp.Function('N'),sp.Function('S'),sp.Function('P')
N0=1-u**(-h)
N=N0+s*Nf(u); sig=1+s*Sf(u); phi0=c*sp.log(u); phi=phi0+s*Pf(u)
# U1 effective field derivative, absorbing gU/kappa
cf=sp.sqrt(2*(z-1)*h)
f0p=cf*u**(a+z)
fprime=f0p*(1+s*(Sf(u)-lamU*Pf(u)))  # fixed Maxwell flux
# metric
t,x=sp.symbols('t x'); coords=[t,u,x]; n=3
gdiag=[-u**(2*a+2*z)*sig**2*N, u**(2*a-2)/N, u**(2*a+2)]
ginv=[1/g for g in gdiag]
Gam=[[[sp.S(0) for _ in range(n)] for _ in range(n)] for _ in range(n)]
for A in range(n):
 for B in range(n):
  for Cc in range(n):
   vv=0
   for Dd in range(n):
    gad=ginv[A] if A==Dd else 0
    if gad==0: continue
    gdc=gdiag[Dd] if Dd==Cc else 0
    gdb=gdiag[Dd] if Dd==B else 0
    gbc=gdiag[B] if B==Cc else 0
    vv += gad*(sp.diff(gdc,coords[B])+sp.diff(gdb,coords[Cc])-sp.diff(gbc,coords[Dd]))/2
   Gam[A][B][Cc]=vv

def Ric(A,B):
 vv=0
 for Cc in range(n):
  vv += sp.diff(Gam[Cc][A][B],coords[Cc])-sp.diff(Gam[Cc][A][Cc],coords[B])
  for E in range(n): vv += Gam[Cc][Cc][E]*Gam[E][A][B]-Gam[Cc][B][E]*Gam[E][A][Cc]
 return vv

def lin(e): return sp.diff(e,s).subs(s,0)
# U1 F contractions. only F_ut=fprime
# F^2=2 f'^2 g^uu g^tt
F2=2*fprime**2*ginv[1]*ginv[0]
# F_{mu sigma} F_nu^sigma: tt=f'^2 g^uu; uu=f'^2 g^tt; xx=0
FF=[fprime**2*ginv[1], fprime**2*ginv[0], sp.S(0)]
Ls=[]; Bgs=[]
for A in range(n):
 dp=sp.diff(phi,coords[A])
 Wu=Ric(A,A)-sp.Rational(1,2)*dp*dp+V0*gdiag[A]*sp.exp(gamma*phi)
 Wu -= sp.Rational(1,2)*sp.exp(lamU*phi)*(FF[A]-sp.Rational(1,2)*gdiag[A]*F2)
 Bgs.append(sp.simplify(Wu.subs(s,0)))
 Ls.append(sp.factor(lin(Wu)))
sqg=sp.sqrt(-sp.prod(gdiag)); kin=sp.diff(sqg*ginv[1]*sp.diff(phi,u),u)/sqg
X=kin+V0*gamma*sp.exp(gamma*phi)-sp.Rational(1,4)*lamU*sp.exp(lamU*phi)*F2
Bgsc=sp.simplify(X.subs(s,0)); Lsc=sp.factor(lin(X))
print('background tt,uu,xx,sc simplified:')
for q in Bgs+[Bgsc]: print(sp.factor(q))
# save expressions in pickle-ish repr and structure coefficients
syms=[sp.diff(Nf(u),u,2),sp.diff(Sf(u),u,2),sp.diff(Pf(u),u,2),sp.diff(Nf(u),u),sp.diff(Sf(u),u),sp.diff(Pf(u),u),Nf(u),Sf(u),Pf(u)]
names=['Npp','Spp','Ppp','Np','Sp','Pp','N','S','P']
for nm,expr in zip(['Ltt','Luu','Lxx','Lsc'],Ls+[Lsc]):
 print('\n'+nm)
 for name,sy in zip(names,syms):
  cc=sp.factor(sp.diff(expr,sy))
  if cc!=0: print(name,sp.sstr(cc))
 rem=expr-sum(sp.diff(expr,sy)*sy for sy in syms)
 print('rem',sp.factor(rem))
# dump via srepr? use sympy pickle
import pickle
with open('/mnt/data/d3_general_operator.pkl','wb') as f: pickle.dump((a,z,u,Ls,Lsc),f)
