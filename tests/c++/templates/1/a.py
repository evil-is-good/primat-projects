f = open('a-uber.h', 'w')
for i in range(1000):
    f.write('template <typename T> T foo' + str(i) + ' () { return T(); }\n')
f.close()
