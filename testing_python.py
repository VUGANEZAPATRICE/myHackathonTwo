import pandas as pd
# dict1 = {'name':['Roger','amina','Kabera'],
#          'profession':['Electronician','engineer','Teacher'],
#          'age':[10,30,40]}

# df = pd.DataFrame(dict1)

# print(df)
# for i,j in df.iterrows():
#     print(df.ix[:4,])
#     print('===============================')
    

stre = """aa
ab
ae
af
ak
am
an
ar
as
av
ay
az
ba
be
bg
bh
bi
bm
bn
bo
br
bs
ca
ce
ch
co
cr
cs
cu
cv
cy
da
de
dv
dz
ee
el
en
eo
es
et
eu
fa
ff
fi
fj
fo
fr
fy
ga
gd
gl
gn
gu
gv
ha
he
hi
ho
hr
ht
hu
hy
hz
ia
id
ie
ig
ii
ik
io
is
it
iu
ja
jv
ka
kg
ki
kj
kk
kl
km
kn
ko
kr
ks
ku
kv
kw
ky
la
lb
lg
li
ln
lo
lt
lu
lv
mg
mh
mi
mk
ml
mn
mr
ms
mt
my
na
nb
nd
ne
ng
nl
nn
no
nr
nv
ny
oc
oj
om
or
os
pa
pi
pl
ps
pt
qu
rm
rn
ro
ru
rw
sa
sc
sd
se
sg
si
sk
sl
sm
sn
so
sq
sr
ss
st
su
sv
sw
ta
te
tg
th
ti
tk
tl
tn
to
tr
ts
tt
tw
ty
ug
uk
ur
uz
ve
vi
vo
wa
wo
xh
yi
yo
za
zh
zu"""
str1= stre.split(" ") 
print(str1)
lang_code=[]
for lang in str1:
    lang_code.append(lang)
    
print(lang_code)
print(len(lang_code))

# language =[aa
# ab
# ae
# af
# ak
# am
# an
# ar
# as
# av
# ay
# az
# ba
# be
# bg
# bh
# bi
# bm
# bn
# bo
# br
# bs
# ca
# ce
# ch
# co
# cr
# cs
# cu
# cv
# cy
# da
# de
# dv
# dz
# ee
# el
# en
# eo
# es
# et
# eu
# fa
# ff
# fi
# fj
# fo
# fr
# fy
# ga
# gd
# gl
# gn
# gu
# gv
# ha
# he
# hi
# ho
# hr
# ht
# hu
# hy
# hz
# ia
# id
# ie
# ig
# ii
# ik
# io
# is
# it
# iu
# ja
# jv
# ka
# kg
# ki
# kj
# kk
# kl
# km
# kn
# ko
# kr
# ks
# ku
# kv
# kw
# ky
# la
# lb
# lg
# li
# ln
# lo
# lt
# lu
# lv
# mg
# mh
# mi
# mk
# ml
# mn
# mr
# ms
# mt
# my
# na
# nb
# nd
# ne
# ng
# nl
# nn
# no
# nr
# nv
# ny
# oc
# oj
# om
# or
# os
# pa
# pi
# pl
# ps
# pt
# qu
# rm
# rn
# ro
# ru
# rw
# sa
# sc
# sd
# se
# sg
# si
# sk
# sl
# sm
# sn
# so
# sq
# sr
# ss
# st
# su
# sv
# sw
# ta
# te
# tg
# th
# ti
# tk
# tl
# tn
# to
# tr
# ts
# tt
# tw
# ty
# ug
# uk
# ur
# uz
# ve
# vi
# vo
# wa
# wo
# xh
# yi
# yo
# za
# zh
# zu]