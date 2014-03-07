import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
# import matplotlib.lines.Line2D as line2d
import matplotlib.lines as lines
x = np.arange(0., 3., 0.1)
# plt.plot([1,2,3],[1,4,9])
# plt.plot(x, x**2)
# l = lines.Line2D([1.0, 2.0], [1.0, 1.0])
# plt.text(1., 2., r'$\mu=100 \sum_{i=0}^{10} $', color='r', fontproperties=mp.font_manager.FontProperties("Ubuntu Mono"), fontsize=22)
# plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
#         arrowprops=dict(facecolor='black', shrink=0.05),
#         )
# plt.fill([0.,1.,0.],[0.,0.,1.],'r', ec='#000000')
# plt.fill([0.,1.,1.],[1.,1.,0.],'b')


def ploting(file_name, func, linew):
    f = open(file_name, "r")
    ls = f.readlines()
    for l in ls:
        obj = l.split('|')
        func([float(i) for i in obj[0].split()], [float(i) for i in obj[1].split()], obj[2].split()[0], lw = linew)
    f.close()

ploting("area.out", plt.fill, 1)
# plt.linewidth = 15
# ploting("border.out", plt.plot, 15)

# plt.plot([0.,1.,1.],[1.,1.,0.],'g',lw=15)
# f = open("out.file","r")
# ls = f.readlines()
# print ls
# for l in ls:
#     tr = l.split('|')
#     plt.fill([float(i) for i in tr[0].split()], [float(i) for i in tr[1].split()], tr[2].split()[0])
# f.close()
    # print [float(i) for i in l.split('|')[0].split()]
    # print l.split('|')[2].split()[0]
# n = int(f.readline())

# l = [float(i) for i in f.read().split()]
# print l
# plt.plot([0.,1.],[0.,1.],'g')
# plt.setp(l)
# l.draw()
plt.show()


# def foo(*a):
#     print a
#     return 0
#
# foo(10)
# foo()
# foo(1,2)
# foo(1,"sdc")
# foo(10,None)
# import webbrowser
# webbrowser.open_new_tab('http://habrahabr.ru/')

# class AA:
#     def foo(self):
#         print "AA"
# class BB:
#     def foo(self):
#         print "BB"
# 
# a = {1:AA,2:BB}
# print a
# 
# bb = object()
# 
# bb.__class__ = a[1]
# bb.foo()
# bb.__class__ = a[2]
# bb.foo()
