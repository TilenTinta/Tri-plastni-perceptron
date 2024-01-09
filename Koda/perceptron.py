import numpy as np

class Perceptron:
    def __init__(self, inNo, hidNo, outNo, ucniFaktor):
        # Podatki o omrežju
        self.vhodiNo = inNo # stevilo vhodov
        self.skritiNo = hidNo # stevilo skritih plasti
        self.izhodiNo = outNo # stevilo izhodov
        self.ucniFaktor = ucniFaktor # faktor učenja

        # Random določi uteži
        self.uteziIn =  np.random.uniform(-0.5, 0.5, (self.skritiNo, self.vhodiNo))
        self.uteziHid =  np.random.uniform(-0.5, 0.5, (self.izhodiNo, self.skritiNo))

        # Random določi bias-e
        self.biasIn = np.random.uniform(-0.5,0.5,(self.skritiNo, 1))
        self.biasHid = np.random.uniform(-0.5,0.5,(self.izhodiNo, 1))


    ### Delovanje mreže ###
    def FeedForward(self, vhodi):
        # XOR: vhod 1x2 -> transoniraš v 2x1 in enak je skalarni produkt

        # Matrične operacije med vhodi in skrito plastjo #
        skritoIn1 = np.dot(self.uteziIn, vhodi)
        skritoIn2 = skritoIn1 + self.biasIn
        self.skritoOut = self.Sigmoid(skritoIn2)

        # Matrične operacije med skrito plastjo in izhodi #
        izhodIn1 = np.dot(self.uteziHid, self.skritoOut)
        izhodIn2 = np.add(izhodIn1, self.biasHid)
        self.izhodOut = self.Sigmoid(izhodIn2)

        # Izhodi #
        return np.round(self.izhodOut)


    # Sigmoidna funkcija
    def Sigmoid(self, x):
        val = 1 / (1 + np.exp(-x))
        return val
    

    # Sigmoidna funkcija - "odvod / gradient" 
    def SigmoidOdvod(self, x):
        val = x * (1 - x)
        return val
    

    ### Učenje mreže ###
    def Ucenje(self, vhodi, izhodReal):

        # Matrične operacije med vhodi in skrito plastjo #
        skritoIn1 = np.dot(self.uteziIn, vhodi) 
        skritoIn2 = skritoIn1 + self.biasIn
        skritoOut = self.Sigmoid(skritoIn2)

        # Matrične operacije med skrito plastjo in izhodi #
        izhodIn1 = np.dot(self.uteziHid, skritoOut)
        izhodIn2 = np.add(izhodIn1, self.biasHid)
        izhodOut = self.Sigmoid(izhodIn2)


        # Racunanje napake na izhodnih nevronih #
        napakeIzhod = np.subtract(izhodReal, izhodOut)
        # Gradient
        gradientOut1 = self.SigmoidOdvod(izhodOut)
        gradientOut2 = np.multiply(gradientOut1, napakeIzhod)
        gradientOut = np.multiply(gradientOut2, self.ucniFaktor)
        # Delta
        skritoOutTrans = np.transpose(skritoOut)
        uteziHidDelta = np.multiply(gradientOut, skritoOutTrans)
        # Popravi uteži in bias
        self.uteziHid = np.add(self.uteziHid, uteziHidDelta)
        self.biasHid = np.add(self.biasHid, gradientOut)


        # Racunanje napake na skritih nevronih #
        uteziHidTrans = np.transpose(self.uteziHid) 
        napakeSkrite1 = np.dot(uteziHidTrans, napakeIzhod)
        # Gradient
        gradientIn1 = self.SigmoidOdvod(skritoOut)
        gradientIn2 = np.multiply(gradientIn1, napakeSkrite1)
        gradientIn = np.multiply(gradientIn2, self.ucniFaktor)
        # Delta
        vhodiOutTrans = np.transpose(vhodi)
        uteziInDelta = np.multiply(gradientIn, vhodiOutTrans)
        # Popravi uteži in bias
        self.uteziIn = np.add(self.uteziIn, uteziInDelta)
        self.biasIn = np.add(self.biasIn, gradientIn)

