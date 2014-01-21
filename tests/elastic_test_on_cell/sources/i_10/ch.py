Ef = 10.0
Em = 1.0
Vf = 0.28
Vm = 0.20

Mf = Ef / (2.0 * (1.0 + Vf))
Mm = Em / (2.0 * (1.0 + Vm))

Kf = Ef / (3.0 * (1.0 - 2.0 * Vf))
Km = Em / (3.0 * (1.0 - 2.0 * Vm))

Nf = 3 - 4 * Vf
Nm = 3 - 4 * Vm

def E11(c):
	numerator = (4.0 * c * (1.0 - c) * (Vf - Vm)**2 * Mm)
        denominator = ((1.0 - c) * Mm / (Kf + Mf / 3.0) + c * Mm / (Km + Mm / 3.0) + 1.0)
        return c * Ef + (1.0 - c) * Em + numerator / denominator

def V12(c):
	numerator = (c * (1.0 - c) * (Vf - Vm) * Mm * (1 / (Km + Mm / 3) - 1 / (Kf - Mf / 3)))
        denominator = ((1.0 - c) * Mm / (Kf + Mf / 3.0) + c * Mm / (Km + Mm / 3.0) + 1.0)
        return c * Vf + (1.0 - c) * Vm + numerator / denominator

def M12(c):
	return Mm * (Mf * (1.0 + c) + Mm * (1.0 - c)) / (Mf * (1.0 - c) + Mm * (1.0 + c))

def K23(c):
	numerator = c * (Kf - Km + (Mf - Mm) / 3.0) * (Km + 4.0 * Mm / 3.0)
        denominator = (Km + 4.0 * Mm / 3.0) + (1.0 - c) * (Kf - Km + (Mf - Mm) / 3.0) 
	return Km + Mm / 3.0 + numerator / denominator #c / (1.0 / (Kf - Km + (Mf - Mm) / 3.0) + (1.0 - c) / (Km + 4.0 * Mm / 3.0))

def M23(c):
	aux1 = Mf / Mm                              #= 1.0
	aux1a = (aux1 - 1.0) * c                    #= 0.0
	aux1aa = aux1 * Nm + 1.0                    #= 3.0
	aux1ab = aux1aa + aux1a                     #= 3.0
	aux1b = aux1 + Nf                           #= 3.0
	aux1c = (aux1 * Nm - Nf) * c**3             #= 0.0
	aux2 = 3.0 * (1.0 - c)**2 * aux1a * aux1b   #= 0.0
	aux3 = aux1 + Nf + aux1c                    #= 3.0

	#print aux1, aux1a, aux1aa, aux1ab, aux1b, aux1c, aux2, aux3

	A1 = aux2
        A2 = (aux1 * Nm + Nf * Nm - aux1c) 
        A3 = (Nm * aux1a - aux1aa)
	A = A1 + A2 * A3

	B1 = -aux2
	B2 = 0.5 * aux1ab
	B3 = (Nm - 1) * aux1b - 2.0 * aux1c
	B4 = 0.5 * (Nm + 1.0) * aux1a * aux3
	B = 2.0 * (B1 + B2 * B3 + B4)

	C1 = aux2
	C2 = aux1ab * aux3
	C = C1 + C2

	D = B**2 - 4.0 * A * C

	#print A, B, C, D, D**0.5

	return ( (- D**0.5 - B) / (2.0 * A)) * Mm	
                    
def E22(c):
        return (4.0 * M23(c) * K23(c)) / (K23(c) + M23(c) + 4.0 * V12(c)**2.0 * M23(c) * K23(c) / E11(c))

def V23(c):
        return (K23(c) - M23(c) - 4.0 * V12(c)**2.0 * M23(c) * K23(c) / E11(c)) / (K23(c) + M23(c) + 4.0 * V12(c)**2.0 * M23(c) * K23(c) / E11(c))

def V21(c):
	#aux = 4.0 * V12(c)**2.0 * M23(c) * K23(c)
        #return aux / (E11(c) * (K23(c) + M23(c)) + aux)
	return V12(c) / E11(c) * E22(c)

f = open('ch.gpd', 'w')

c = 0
while c < 1.01:
	f.write(str(c) + ' ')
	f.write(str(E22(c)) + ' ')
	f.write(str(V23(c)) + ' ')
	f.write(str(V12(c) / E11(c) * E22(c)) + ' ')
	f.write(str(V23(c)) + ' ')
	f.write(str(E22(c)) + ' ')
	f.write(str(V21(c)) + ' ')
	f.write(str(V12(c)) + ' ')
	f.write(str(V12(c)) + ' ')
	f.write(str(E11(c)) + ' ')
	f.write(str(M23(c)) + ' ')
	f.write(str(M12(c)) + ' ')
	f.write(str(K23(c)) + '\n')
	#f.write(str(c * Ef + (1.0 - c) * Em) + '\n')
	c = c + 0.01

f.close()

print Km, Kf

