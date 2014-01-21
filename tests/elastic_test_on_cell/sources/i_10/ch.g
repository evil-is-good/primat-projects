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

E11_aux1(c) = (4.0 * c * (1.0 - c) * (Vf - Vm)**2 * Mm)
E11_aux2(c) = ((1.0 - c) * Mm / (Kf + Mf / 3.0) + c * Mm / (Km + Mm / 3.0) + 1.0)
E11(c) = c * Ef + (1.0 - c) * Em + E11_aux1(c) / E11_aux2(c)

V12_aux1(c) = (c * (1.0 - c) * (Vf - Vm) * Mm * (1 / (Km + Mm / 3) - 1 / (Kf - Mf / 3)))
V12_aux2(c) = ((1.0 - c) * Mm / (Kf + Mf / 3.0) + c * Mm / (Km + Mm / 3.0) + 1.0)
V12(c) = c * Vf + (1.0 - c) * Vm + V12_aux1(c) / V12_aux2(c)

dM12(c) = Mm * (Mf * (1.0 + c) + Mm * (1.0 - c)) / (Mf * (1.0 - c) + Mm * (1.0 + c))

K23_aux1(c) = c * (Kf - Km + (Mf - Mm) / 3.0) * (Km + 4.0 * Mm / 3.0)
K23_aux2(c) = (Km + 4.0 * Mm / 3.0) + (1.0 - c) * (Kf - Km + (Mf - Mm) / 3.0)
K23(c) = Km + Mm / 3.0 + K23_aux1(c) / K23_aux2(c) 


M23_aux1(c)   = Mf / Mm                     
M23_aux1a(c)  = (M23_aux1(c) - 1.0) * c             
M23_aux1aa(c) = M23_aux1(c) * Nm + 1.0           
M23_aux1ab(c) = M23_aux1aa(c) + M23_aux1a(c)               
M23_aux1b(c)  = M23_aux1(c) + Nf                        
M23_aux1c(c)  = (M23_aux1(c) * Nm - Nf) * c**3             
M23_aux2(c)   = 3.0 * (1.0 - c)**2 * M23_aux1a(c) * M23_aux1b(c)  
M23_aux3(c)   = M23_aux1(c) + Nf + M23_aux1c(c)                  

M23_A1(c) = M23_aux2(c)
M23_A2(c) = (M23_aux1(c) * Nm + Nf * Nm - M23_aux1c(c)) 
M23_A3(c) = (Nm * M23_aux1a(c) - M23_aux1aa(c))
M23_A(c) = M23_A1(c) + M23_A2(c) * M23_A3(c)

M23_B1(c) = -M23_aux2(c)
M23_B2(c) = 0.5 * M23_aux1ab(c)
M23_B3(c) = (Nm - 1) * M23_aux1b(c) - 2.0 * M23_aux1c(c)
M23_B4(c) = 0.5 * (M23_Nm + 1.0) * M23_aux1a(c) * M23_aux3(c)
M23_B(c) = 2.0 * (M23_B1(c) + M23_B2(c) * M23_B3(c) + M23_B4(c))

M23_C1(c) = M23_aux2(c)
M23_C2(c) = M23_aux1ab(c) * M23_aux3(c)
M23_C(c) = M23_C1(c) + M23_C2(c)

M23_D(c) = M23_B(c)**2 - 4.0 * M23_A (c)* M23_C(c)

M23(c) = ( (- M23_D(c)**0.5 - M23_B(c)) / (2.0 * M23_A(c))) * Mm	
                    

E22(c) = (4.0 * M23(c) * K23(c)) / (K23(c) + M23(c) + 4.0 * V12(c)**2.0 * M23(c) * K23(c) / E11(c))

V23(c) = (K23(c) - M23(c) - 4.0 * V12(c)**2.0 * M23(c) * K23(c) / E11(c)) / (K23(c) + M23(c) + 4.0 * V12(c)**2.0 * M23(c) * K23(c) / E11(c))

V21(c) = V12(c) / E11(c) * E22(c)

print E11(0.04)

  
