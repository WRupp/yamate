
import copy

import numpy as np
from yamate.utils import mathroutines as mr
from yamate.materials import material

class Properties:

    names = [
    "mu", "nu", "Bulk",
    "kfa", "kc", "keta", "Sy0", "kh", 
    "s0", "scv", "sg", "sz", "sb",
    "knh", "kHiso", "kcH", "FlagHardening", 
    "knd", "km", "kR", "kg", "kS", "kN", "threshold",
    "FlagPlasDam", "FlagHidrDam", "params", "alpha_guess"]

    def __init__(self,
        mu=None, nu=None, Bulk=None,
        kfa=None, kc=None, keta=None, Sy0=None, kh=None, 
        s0=None, scv=None, sg=None, sz=None, sb=None,
        knh=None, kHiso=None, kcH=None, FlagHardening=1, 
        knd=None, km=None, kR=None, kg=None, kS=None, kN=None, threshold=None,
        FlagPlasDam=1, FlagHidrDam=1, params = np.ones(3), alpha_guess=0.0
        ):
        # Hyperelastic
        self.mu = mu
        self.nu = nu
        self.Bulk = Bulk

        #Viscoplastic
        self.kfa = kfa
        self.kc = kc
        self.keta = keta
        self.Sy0 = Sy0
        self.kh = kh
        
        self.s0 = s0
        self.scv = scv
        self.sg = sg
        self.sz = sz
        self.sb = sb

        # Isotropic Hardening
        self.FlagHardening = FlagHardening        
        self.knh = knh
        self.kHiso = kHiso
        self.kcH = kcH
                
        # Damage         
        self.FlagPlasDam = FlagPlasDam
        self.threshold = threshold 
        self.kS = kS
        self.kN = kN

        self.FlagHidrDam = FlagHidrDam
        self.knd = knd
        self.km = km
        self.kR = kR
        self.kg = kg 

        # Numerical Integration Parameters
        self.params = params
        # Tolerance Local-Newton Parameters
        self.alpha_guess = alpha_guess


class VariationalViscoHydrolysis(material.Material):

    name = "variational_visco_hydro"

    def __init__(self, props={}):
        self.state.Fpn = np.eye(3)

        self.state.vin = np.zeros(10)
        self.state.vin[0] = 1.0

        self.state.timen = 0.0

        self.properties = Properties(**props)

    def hencky(self, props, etr, Ei):

        eps = np.zeros(3)
        G = props.mu

        if ( etr.shape == (3,) ) :
            eps = np.array([ etr[0] , etr[1], etr[2]])
        else :
            eps[0] = mr.tensor_inner_product(etr,Ei[:,:,0])
            eps[1] = mr.tensor_inner_product(etr,Ei[:,:,1])
            eps[2] = mr.tensor_inner_product(etr,Ei[:,:,2])

        vdWtr = 2 * G * eps

        dWtr = np.eye(3)
        dWtr[0,0] = vdWtr[0]
        dWtr[1,1] = vdWtr[1]
        dWtr[2,2] = vdWtr[2]

        d2Wtr = 2*G * np.eye(3)

        energye = G*(eps[0]**2 + eps[1]**2 + eps[2]**2)
        
        return dWtr, d2Wtr, energye

    def kappa_functions(self, props, alpha):

        if props.FlagHardening == 1 :
            kappa = props.kHiso * alpha
            dkappa = props.kHiso
            energyp = 0.5 * props.kHiso * (alpha ** 2.0)
        elif props.FlagHardening == 2 :
            kappa = props.kHiso * ( 1.0 - np.exp(-props.knh*alpha) ) + props.kcH * alpha
            dkappa = props.kHiso * props.knh * np.exp(-props.knh * alpha) + props.kcH
            energyp = props.kHiso * alpha + ( props.kHiso * (np.exp(-props.knh*alpha) - 1)) / props.knh + 0.5 * props.kcH * alpha**2.0
        elif props.FlagHardening == 3:
            kappa = props.kHiso * (np.exp(props.knh * alpha - 1))
            dkappa = props.kHiso * props.knh * (np.exp(props.knh * alpha) )
            energyp = props.kHiso* ((np.exp(props.knh * alpha - 1) / props.knh - alpha))
        else:
            raise Exception("FlagHardening is not correctly defined")
        
        return kappa, dkappa, energyp
        
    def vol_functions(self, props, J):

        # G = props.mu
        Bulk = props.Bulk

        dUdJ = (1.0/J)*(Bulk)*np.log(J)
        energyv = 0.5 * (Bulk) * ( np.log(J) ** 2.0 )
        return dUdJ, energyv

    def exp_matrix_sym_3x3(self, M):

        eigenvalues, eigenvectors = np.linalg.eigh( M )

        expM = 0.0e0
        for k in range(3):
            expM = expM + np.exp(eigenvalues[k]) * mr.tensor_product(eigenvectors[:,k], eigenvectors[:,k])

        return expM

    def visco_arrasto(self, props, alpha):

        s0 = props.s0
        scv = props.scv 
        sg = props.sg 
        sz = props.sz 
        sb = props.sb 
        R = alpha 

        fArr = scv + np.exp(-sz*R) * ( (s0-scv)*np.cosh(sb*R)+sg*np.sinh(sb*R))

        c1 = (s0-scv)*sb - sg*sz 
        c2 = sg*sb - (s0-scv)*sz 
        c3 = c1 * sb - sz * c2 
        c4 = c2 * sb - sz * c1 
            
        dfArr = np.exp(-sz*R)*(c1*np.sinh(sb*R)+c2*np.cosh(sb*R))
        d2fArr = np.exp(-sz*R) * (c3*np.cosh(sb*R) + c4*np.sinh(sb*R))

        return fArr, dfArr, d2fArr

    def hydro_func(self, props, vin, vin1, deltat):

        Sy0 = props.Sy0
        m  = props.km
        n  = props.knd
        kR = props.kR
        g  = props.kg
        kS = props.kS
        kN = props.kN
        theta = props.params[0]
        gamma = props.params[1]
        # zeta  = props.params[2]

        dpn     = vin[1]
        dhn     = vin[2]
        alphan  = vin[3]
        Yn      = vin[4]

        # dpn1    = pvin1[1]
        dhn1    = vin1[2]
        alphan1 = vin1[3]
        Yn1     = vin1[4]

        # Ddp = dpn1-dpn
        Ddh = dhn1-dhn
        delta_alpha = alphan1-alphan

        dn = dpn + dhn
        # dn1 = dn + (Ddp + Ddh)

        Ytheta = (1-theta)*Yn + theta*Yn1
        Ygamma = (1-gamma)*Yn + gamma*Yn1
        # Yzeta = (1-zeta)*Yn + zeta*Yn1

        dtheta = dn + theta*((delta_alpha*(Ytheta**kS)/kN) + Ddh)   

        TERM1= -(Yn1+g) + ( kR / (((1-dtheta)**n) * ((Ygamma+g)**(m-1)))) * (Ddh/deltat)
        TERM2 = theta*deltat * ( ( n*kR / (2*( (1-dtheta)**(n+1) )*( (Ygamma+g)**(m-1) )) ) * ( (Ddh/deltat)**2))
        TERM3 = deltat*(-Sy0*delta_alpha/deltat)

        VARS = TERM1 + TERM2 + TERM3

        return VARS

    def compute_expressions(self, props, vin, vin1, deltat):

        Sy0 = props.Sy0
        m  = props.km
        n  = props.knd
        kR = props.kR
        g  = props.kg
        kS = props.kS
        kN = props.kN
        keta = props.keta
        kc = props.kc
        theta = props.params[0]
        gamma = props.params[1]

        dpn     = vin[1]
        dhn     = vin[2]
        alphan  = vin[3]
        Yn      = vin[4]

        dpn1    = vin1[1]
        dhn1    = vin1[2]
        alphan1 = vin1[3]
        Yn1     = vin1[4]

        Ddp = dpn1-dpn
        Ddh = dhn1-dhn
        delta_alpha = alphan1-alphan

        dn = dpn + dhn
        dn1 = dn + (Ddp + Ddh)

        Ytheta = (1-theta)*Yn + theta*Yn1
        Ygamma = (1-gamma)*Yn + gamma*Yn1

        dtheta = dn + theta*((delta_alpha*(Ytheta**kS)/kN) + Ddh)  

        fArr, dfArr, _ = self.visco_arrasto(props, alphan1)
        kappa, _, _ = self.kappa_functions(props, alphan1)

        FATOR = (kR/2.0) * (Ddh/deltat)**2.0

        FG = ( (1-dn1)  
        + deltat * (delta_alpha/deltat) * ((Yn1**(kS))/kN) 
        - deltat * Sy0*(delta_alpha/deltat)*kS*delta_alpha*((Yn1**(kS-1.0))/kN)
        + deltat * FATOR * gamma * (1.0-m)/( ((1.0-dtheta)**(n))*(Ygamma+g)**(m)) 
        + deltat * FATOR * theta * n/( ((1.0-dtheta)**(n+1))*(Ygamma+g)**(m-1.0)) 
        * theta*kS*delta_alpha*((Ytheta**(kS-1.0))/kN)
        )

        FA = FG*kappa + (1-dn1)*Sy0 + fArr*(delta_alpha/(deltat*kc))**(keta)

        FB = ( (kc/(keta+1))*((delta_alpha/(deltat*kc))**(keta+1))*dfArr
        - Sy0*(delta_alpha/deltat)*((Yn1**(kS))/kN)
        +  FATOR * theta * n/( ((1-dtheta)**(n+1))*(Ygamma+g)**(m-1))*((Ytheta**(kS))/kN)
        )

        return FA, FB, FG

    def rm_functions(self, dWede , M , props, vin, vin1, deltat):

        VFun = np.empty(2)

        FA, FB, FG = self.compute_expressions(props, vin, vin1, deltat)

        Seq = mr.tensor_inner_product(dWede,M)
        VFun[0] = -FG * Seq + FA + deltat * FB
        Vars = self.hydro_func(props, vin, vin1, deltat)
        VFun[1] = Vars

        return VFun

    def compute_hydrolytic(self, props,vin,vin1,deltat):
        
        m  = props.km
        n  = props.knd
        kR = props.kR
        g  = props.kg
        kS = props.kS
        kN = props.kN
        theta = props.params[0]
        gamma = props.params[1]

        pvin=vin.copy()
        pvin1=vin1.copy()

        dpn     = pvin[1]
        dhn     = pvin[2]
        alphan  = pvin[3]
        Yn      = pvin[4]

        dpn1    = pvin1[1]
        dhn1    = pvin1[2]
        alphan1 = pvin1[3]
        Yn1     = pvin1[4]

        Ddp = dpn1-dpn
        Ddh = dhn1-dhn
        delta_alpha = alphan1-alphan

        dn = dpn + dhn

        Ygamma = (1-gamma)*Yn + gamma*Yn1
        Ytheta = (1-theta)*Yn + theta*Yn1

        Ddh   =  (( ( ((1-dn)**n) * ((Ygamma+g)**(m)) ) ) /kR ) * deltat

        dn1 = dn + (Ddp + Ddh)

        dtheta = dn + theta*((delta_alpha*(Ytheta**kS)/kN) + Ddh)

        DELTA = 0.0

        pvin1[0] = 1 - dn1
        pvin1[2] = dhn + Ddh

        FVAL = self.hydro_func(props, pvin, pvin1, deltat)

        erro = 1.0
        TOL = 1.0e-6
        cont = 0

        while (erro > TOL):

            Kdh =  ( 
                + (kR/( ((1-dtheta)**n) * (Ygamma+g)**(m-1)) ) * ((1/deltat) + theta*(n/(1-dtheta))*(Ddh/deltat))
                + theta* (n*kR/( ((1-dtheta)**(n+2)) * (Ygamma+g)**(m-1)) ) * ( (1-dtheta)*(Ddh/deltat) +theta*deltat*((n+1)/2)*((Ddh/deltat)**2))
            )

            DELTA = - FVAL / Kdh

            Ddh = Ddh + DELTA

            pvin1[2] = pvin[2] + Ddh

            FVAL = self.hydro_func(props, pvin, pvin1, deltat)

            erro = abs(FVAL)

            cont=cont+1

            if  ( (cont > 20) or (Ddh < 0.0) ) :
                print('compute_hydrolytic: Your circuit`s dead, there`s something wrong. Can you hear me, Major Tom?')
                quit()
                return
                
        VARS = Ddh

        return VARS

    def resid_functions(self, epstr, M, Ea, props, J, vin, vin1, deltat, delta_alpha, Ddh):

        # dWede = np.zeros((3,3))
        # dWedej = np.zeros((3,3))
        # dWe2de2j= np.zeros((3,3))

        eps= np.zeros((3,3))
        dummy= np.zeros((3))

        kS = props.kS
        kN = props.kN
        zeta  = props.params[2]

        pvin1=vin1.copy()
        pvin=vin.copy()

        dpn     = pvin[1]
        dhn     = pvin[2]
        alphan  = pvin[3]
        Yn      = pvin[4]

        dpn1    = pvin1[1]
        dhn1    = pvin1[2]
        alphan1 = pvin1[3]
        Yn1     = pvin1[4]

        Ddp = dpn1-dpn

        alphan1 = alphan + delta_alpha
        eps = epstr - delta_alpha*M

        dWedej, _, energye = self.hencky(props, eps, Ea)
        dummy[0], dummy[1], energyp = self.kappa_functions(props, alphan1)
        dummy[0], energyv = self.vol_functions(props, J)

        Yn1 = energye + energyv + energyp
        Yzeta = (1-zeta)*Yn+zeta*Yn1

        Ddp=delta_alpha*(Yzeta**kS)/kN

        dpn1=dpn+Ddp
        dhn1=dhn+Ddh

        pvin1[1] = dpn1
        pvin1[2] = dhn1
        pvin1[3] = alphan1
        pvin1[4] = Yn1

        dWede= ( 
        + dWedej[0,0]*Ea[:,:,0] 
        + dWedej[1,1]*Ea[:,:,1]
        + dWedej[2,2]*Ea[:,:,2]
        )

        VFun = self.rm_functions(dWede, M , props, pvin, pvin1, deltat)

        return VFun

    def fixed_point_search(self, epstr, M, Ea, props, J, vin, vin1, deltat, DELTA, flag_where):

        dummy = np.zeros(3)

        kS = props.kS
        kN = props.kN
        zeta  = props.params[2]

        TOL = 1.0e-6

        pvin1=vin1.copy()
        pvin=vin.copy()

        dpn     = pvin[1]
        dhn     = pvin[2]
        alphan  = pvin[3]
        Yn      = pvin[4]

        dpn1    = pvin1[1]
        dhn1    = pvin1[2]
        alphan1 = pvin1[3]
        Yn1     = pvin1[4]

        Ddp = dpn1-dpn
        Ddh = dhn1-dhn
        delta_alpha = DELTA[0]

        erro = 1
        cont = 1
        conti = 1

        delta_alpha0=delta_alpha

        VFun = self.resid_functions(epstr, M, Ea, props, J, vin, vin1, deltat, delta_alpha, Ddh)

        while ((VFun[0] > 0.0e0) and (delta_alpha >= 1.0e16)):
            delta_alpha=delta_alpha*1.0e-1
            VFun = self.resid_functions(epstr, M, Ea, props, J, vin, vin1, deltat, delta_alpha, Ddh)
            delta_alpha0=delta_alpha

        if ((VFun[0] > 0.0e0) and (abs(delta_alpha) <= 1.0e-16)):
            delta_alpha =1.0e-16
        else:
            while ((erro > TOL) and (cont < 20)):
                fator = 1
                # Search for a positive residue
                while (VFun[0] < 0.0e0):
                    delta_alpha=delta_alpha0*((10)**fator)
                    VFun = self.resid_functions(epstr, M, Ea, props, J, vin, vin1, deltat, delta_alpha, Ddh)
                    fator=fator+1

                a=delta_alpha0
                b=delta_alpha
                c=0.5e0*(a+b)

                flag_restart = 1
                conti = 1

                # ! BEGIN - Bissection Method - Finds delta_alpha with fixed Ddh
                while (flag_restart == 1):
                    VFun = self.resid_functions(epstr, M, Ea, props, J, vin, vin1, deltat, c, Ddh)

                    if (VFun[0] < 0.0e0):
                        a = c
                    else:
                        b = c
                    
                    if (abs(VFun[0]) <= TOL) :
                        flag_restart = 0
                    else:
                        conti=conti+1
                        if ((0.5e0*abs(a-b) < 1.0e-16) or (conti >= 50)):
                            if (conti>=50):
                                print("Bissection method error")
                                exit 
                            else:
                                VFun = self.resid_functions(epstr, M, Ea, props, J, vin, vin1, deltat, a, Ddh)
                                DELTA = [a, Ddh]
                                return DELTA
                        else:
                            c=0.5e0*(a+b)
                # ! END - BISSECTION METHOD

                # ! BEGIN - Newton's method - Search for Ddh with fixed delta_alpha 
                delta_alpha = c

                alphan1 = alphan + delta_alpha
                eps = epstr - delta_alpha*M

                _, _, energye = self.hencky(props, eps, Ea)
                dummy[0], dummy[1], energyp = self.kappa_functions(props, alphan1)
                dummy[0], energyv = self.vol_functions(props, J)
                Yn1 = energye + energyv + energyp
                Yzeta = (1-zeta)*Yn+zeta*Yn1

                Ddp=delta_alpha*(Yzeta**kS)/kN

                dpn1=dpn+Ddp
                dhn1=dhn+Ddh

                pvin1[1] = dpn1
                pvin1[2] = dhn1
                pvin1[3] = alphan1
                pvin1[4] = Yn1

                Ddh = self.compute_hydrolytic(props, pvin, pvin1, deltat)

                VFun = self.resid_functions(epstr, M, Ea, props, J, vin, vin1, deltat, c, Ddh)

                erro = mr.norm(VFun)

                cont = cont + 1

                if ((delta_alpha < 1.0e-16) or  (Ddh < 0.0e0) or (cont > 20)):
                    print('ERROR')
                    return
                
        DELTA = [delta_alpha, Ddh]

        return DELTA

    def return_mapping(self, etr, Ea, M, J, props, vin, vin1, deltat, flag_where):

        dummy = np.zeros(3)
        VARS = np.empty(4)

        kS = props.kS
        kN = props.kN
        zeta  = props.params[2]
        pvin6 = vin[5] 


        if (pvin6 == 0 ):
            alpha_guess = props.alpha_guess
        else:
            alpha_guess = pvin6 * 1.0e-3
            if (alpha_guess < 1.0e-16):
                alpha_guess = 1.0e-16

        DELTA = 0.0e0
        # !===========================================================
        # !===========================================================
        vetr = etr.copy()

        pvin1=vin1.copy()
        pvin=vin.copy()

        dpn     = pvin[1]
        dhn     = pvin[2]
        alphan  = pvin[3]
        Yn      = pvin[4]

        dpn1    = pvin1[1]
        dhn1    = pvin1[2]
        alphan1 = pvin1[3]
        Yn1     = pvin1[4]

        Ddp = dpn1-dpn
        Ddh = dhn1-dhn
        Ddh = 0
        delta_alpha = alpha_guess

        alphan1 = 0.0e0
        alphan1 = alphan+delta_alpha

        dWedej, _, dummy[0] = self.hencky(props, vetr, Ea)

        dWede  = (
            +  dWedej[0,0]*Ea[:,:,0]
            +  dWedej[1,1]*Ea[:,:,1]
            +  dWedej[2,2]*Ea[:,:,2]
        )

        epstr=0.e0
        for k in range(3):
            epstr = epstr + etr[k]*Ea[:,:,k] #,0

        DELTA = [delta_alpha, Ddh]

        DELTA = self.fixed_point_search(epstr, M, Ea, props, J, vin, vin1, deltat, DELTA, flag_where)

        delta_alpha = DELTA[0]
        Ddh = DELTA[1]

        alphan1 = alphan + delta_alpha

        eps = epstr - delta_alpha*M

        dWedej, _, energye = self.hencky(props, eps, Ea)
        dummy[0], dummy[1], energyp = self.kappa_functions(props, alphan1)
        dummy[0], energyv = self.vol_functions(props, J)
        Yn1 = energye + energyv + energyp
        Yzeta = (1-zeta)*Yn+zeta*Yn1

        Ddp=delta_alpha*(Yzeta**kS)/kN

        dpn1=dpn+Ddp
        dhn1=dhn+Ddh

        dWede  = (
            +  dWedej[0,0]*Ea[:,:,0]
            +  dWedej[1,1]*Ea[:,:,1]
            +  dWedej[2,2]*Ea[:,:,2]
        )

        VARS[0]=alphan1
        VARS[1]=Ddp
        VARS[2]=Ddh
        VARS[3]=Yn1

        return VARS, dWede





class VariationalViscoHydrolysisAxi(VariationalViscoHydrolysis):

    def calculate_state(self, F, time=None, **kwargs):
        
        trial_state = copy.deepcopy(self.state)

        if time == 0.0: return trial_state
        
        trial_state.F = copy.deepcopy(F)
        Fn1 = copy.deepcopy(F)

        I = np.eye(3)

        ctr=np.empty((3))

        vdWede = np.empty((3))
        dWede = np.empty((3,3))
        etr = np.empty((3,))

        # ! -----------------------------------------------------------------------------------

        if ( np.isnan(Fn1[0,0]) ) :
            print('Fn1[0,0] is NAN')
            trial_state.cauchy_stress[:] = -np.log(0.0)            
            trial_state.error = True
            return trial_state

        vin  = copy.deepcopy( self.state.vin)
        vin1 = copy.deepcopy( self.state.vin)

        dn   = 1.0e0 - vin[0]
        dpn  = vin[1]
        dhn  = vin[2]
        alphan = vin[3]

        timen = copy.deepcopy( self.state.timen)
        Fpn = copy.deepcopy( self.state.Fpn)

        timen1 = time

        deltat = timen1 - timen
        assert deltat != 0.0

        Sy0  = self.properties.Sy0
        
        J = np.linalg.det(Fn1)
        Cn1 = np.matmul(np.transpose(Fn1), Fn1) 
        Fn1_iso = (J**(-1.0e0/3.0e0))*Fn1
        Cn1_iso = np.matmul(np.transpose(Fn1_iso), Fn1_iso)
        Fpn_inv = np.linalg.inv(Fpn)
        Ctr_iso= np.matmul( 
            np.transpose(Fpn_inv),
            np.matmul(Cn1_iso, Fpn_inv)
        )
        
        eigenvalues, eigenvectors = np.linalg.eigh(Ctr_iso)
        ctr = eigenvalues[:]

        Ea = np.empty(shape=(3,3,3))
        for k in range(3):
            Ea[:,:,k] = mr.tensor_product( eigenvectors[:,k], eigenvectors[:,k])

        etr = 0.5e0*np.log(eigenvalues)

        dWtrj, _, energye = self.hencky(self.properties, etr, Ea)
        kappa, _, energyp = self.kappa_functions(self.properties, alphan)
        dUdJ, energyv = self.vol_functions(self.properties, J)

        Yn1 = energye + energyv + energyp 
        
        vin1[4] = Yn1

        Ddh = self.compute_hydrolytic(self.properties, vin, vin1, deltat)

        Ddp = 0
        dn1 = dn + (Ddp + Ddh)

        delta_alpha=0

        dhn1    = dhn + Ddh
        vin1[0] = 1.0e0-dn1
        vin1[2] = dhn1
        vin1[3] = vin[3]

        if (np.isnan( vin1[2])) :
            print('vin1(3) is NAN')
            trial_state.cauchy_stress[:] = -np.log(0.0)            
            trial_state.error = True
            return trial_state

        FA, FB, FG = self.compute_expressions(self.properties, vin, vin1, deltat)

        dWtr = (
            + dWtrj[0,0]*Ea[:,:,0]
            + dWtrj[1,1]*Ea[:,:,1]
            + dWtrj[2,2]*Ea[:,:,2]
        )

        devdWtr = mr.deviatoric(dWtr)

        norma=np.sqrt((mr.tensor_inner_product(dWtr, dWtr)))

        if (norma == 0):
            M = I
        else:
            norma=np.sqrt(mr.tensor_inner_product(devdWtr, devdWtr))
            M = np.sqrt(3.0e0/2.0e0)*devdWtr/norma
            
        Ttrial = mr.tensor_inner_product(dWtr, M)

        finelast =  kappa + ((1-dn1)*Sy0 + deltat*FB )/FG
        finelast2 = (FA + deltat*FB)/FG
        
        ratio = abs(finelast/finelast2)
        if (abs(ratio-1.0e0) > 1.0e-8) :
            print('finelast ratio has found a problem')
            raise ValueError

        ftrial = - Ttrial + finelast
        qfunc = ftrial/finelast

        TOLESC = 1.0e-4

        if ( (qfunc) >= -TOLESC):
            trial_state.flag_ELAST = 1

            if (self.properties.FlagHidrDam == 0):
                Ddh=0.0e0
        
            wn1=(1.0e0-dn1)
            dpn1=dpn
            alphan1=alphan

            dWdCtr = 0.0e0
            for k in range(3):
                dWdCtr = dWdCtr + (0.5e0*dWtrj[k,k]/ctr[k]) * Ea[:,:,k]
            

            dWdC = np.matmul( Fpn_inv, np.matmul( dWdCtr, np.transpose(Fpn_inv) ) )

            DEV_dWdC = dWdC - (1.0e0/3.0e0)*mr.tensor_inner_product(dWdC, Cn1)* np.linalg.inv(Cn1)

            dUdJ, energyv = self.vol_functions(self.properties, J)

            stress0 = 2.0e0*(J**(-2.0e0/3.0e0))*DEV_dWdC + J * dUdJ * np.linalg.inv(Cn1)

            stress =  FG*stress0

            # Modified Cauchy Stress - Calculated in 3D Tensorial Format and converted to Voigt notation.
            Sn1 = np.matmul(np.matmul(Fn1,stress),np.transpose(Fn1))/J

            # saving stress and internal variables to a trial state
            trial_state.cauchy_stress = mr.to_voigt(Sn1)

            fArrn1, _, _ = self.visco_arrasto(self.properties, alphan1)

            trial_state.Fpn = Fpn
            trial_state.vin = [wn1, dpn1, dhn1, alphan1, Yn1, 0, 0, 0, 0, 0]
            trial_state.dWdCiso = dWdC
            trial_state.DEV_dWdCiso = DEV_dWdC
        
        else:
            trial_state.flag_ELAST = 0

            vin1[7]=qfunc
            VARS, dWede = self.return_mapping(etr, Ea, M, J, self.properties, vin, vin1, deltat, 1)
            
            alphan1 = VARS[0]
            Ddp     = VARS[1]
            Ddh     = VARS[2]
            Yn1     = VARS[3]

            delta_alpha = alphan1 - alphan

            dhn1=dhn+Ddh
            dpn1=dpn+Ddp
            dn1=dn+(Ddp+Ddh)

            alphaM = delta_alpha * M

            expM = self.exp_matrix_sym_3x3(alphaM)

            Fpn1 = np.matmul(expM, Fpn)

            vdWede[0] = mr.tensor_inner_product(dWede, Ea[:,:,0])
            vdWede[1] = mr.tensor_inner_product(dWede, Ea[:,:,1])
            vdWede[2] = mr.tensor_inner_product(dWede, Ea[:,:,2])

            dWdCtr = 0.0e0
            for k in range(3):
                dWdCtr = dWdCtr + (0.5e0*vdWede[k]/ctr[k]) * Ea[:,:,k]

            dWdC = np.matmul( Fpn_inv, np.matmul(dWdCtr, np.transpose(Fpn_inv)))

            DEV_dWdC = dWdC - (1.0e0/3.0e0)*mr.tensor_inner_product(dWdC, Cn1)* np.linalg.inv(Cn1)

            wn1=(1.0e0 - dn1)

            dUdJ, energyv = self.vol_functions(self.properties, J)

            vin1 = [wn1, dpn1, dhn1, alphan1, Yn1, 0, 0, 0, 0, 0] #ver se nao dá problema por ser lista

            FA, FB, FG = self.compute_expressions(self.properties, vin, vin1, deltat)

            stress0 = 2.0e0*(J**(-2.0e0/3.0e0))*DEV_dWdC + J * dUdJ * np.linalg.inv(Cn1)

            stress = FG*stress0

            # Modified Cauchy Stress - Calculated in 3D Tensorial Format and converted to Voigt notation.
            Sn1 = np.matmul(np.matmul(Fn1,stress), np.transpose(Fn1))/J

            # saving stress and internal variables to a trial state
            trial_state.cauchy_stress = mr.to_voigt(Sn1)

            fArrn1, _, _ = self.visco_arrasto(self.properties, alphan1)

            trial_state.Fpn =Fpn1
            trial_state.vin = [wn1, dpn1, dhn1, alphan1, Yn1, delta_alpha, fArrn1, 0, 0, 0]
            trial_state.dWdCiso = dWdC
            trial_state.DEV_dWdCiso = DEV_dWdC

        return trial_state