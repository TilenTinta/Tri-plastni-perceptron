import numpy as np
#import matplotlib.pyplot as plt
import random as rnd
from multiprocessing import Pool
#import os
import torch
from torch import nn, optim

import perceptron
import perceptronPytorch

if __name__ == "__main__":

    ############################
    ##### PERCEPTRON - MOJ #####
    ############################

    """
    ###---- XOR PROBLEM ----###

    # Zbirka podatkov
    ucniInData = np.array([[0,0],[1,0],[0,1],[1,1]]) # [[0,0],[1,0],[0,1],[1,1]]
    ucniOutData = np.array([[0],[1],[1],[0]]) # [[0],[1],[1],[0]]

    # Nastavljanje podatkov o mneži (3 plastni perceptron) 
    steviloVhodov = 2
    steviloSkritih = 2
    steviloIzhodov = 1
    ucniFaktor = 0.06

    perc = perceptron.Perceptron(steviloVhodov, steviloSkritih, steviloIzhodov, ucniFaktor)

    ### Program ###
    # Učenje #
    for i in range(100000):
        izberi = rnd.sample(range(0,4), 1) # drugače se ne nauči pravilno
        perc.Ucenje(ucniInData[izberi].reshape(-1, 1),ucniOutData[izberi].reshape(1, 1))

    # Test #
    testData = np.array([[0,0]])
    rezultat = perc.FeedForward(testData.reshape(-1, 1))
    print("Test1: pricakovano: 0, rezultat:", rezultat)

    testData = np.array([[0,1]])
    rezultat = perc.FeedForward(testData.reshape(-1, 1))
    print("Test2: pricakovano: 1, rezultat:", rezultat)

    testData = np.array([[1,0]])
    rezultat = perc.FeedForward(testData.reshape(-1, 1))
    print("Test3: pricakovano: 1, rezultat:", rezultat)

    testData = np.array([[1,1]])
    rezultat = perc.FeedForward(testData.reshape(-1, 1))
    print("Test4: pricakovano: 0, rezultat:", rezultat)
    """


    
    ###---- ISOLET PROBLEM ----###

    # Zbirka podatkov
    ucniPodatki = np.genfromtxt(".\Tri-plastni-perceptron\Koda\isolet1+2+3+4.data", delimiter=', ', dtype=float)
    testniPodatki = np.genfromtxt(".\Tri-plastni-perceptron\Koda\isolet5.data", delimiter=', ', dtype=float)
    #print(ucniPodatki[0])

    # Nastavljanje podatkov o mneži (3 plastni perceptron)
    steviloVhodov = len(ucniPodatki[0]) - 1
    steviloSkritih = 300
    steviloIzhodov = 26
    ucniFaktor = 0.05
    epohe = 100

    perc = perceptron.Perceptron(steviloVhodov, steviloSkritih, steviloIzhodov, ucniFaktor)

    # Rezultati testov:
    # ! Brez epoh, random izbrani vzorci in vsak ponovljen nekaj krat - pustil ker sem že naredu ampak ne upoštevam !
    # 1) 300 skritih, 0.1 faktor, 100000 iteracij (random, 5 krat vsak vzorec): 
    #   - Testni vzorci - Pravilnih: 1437 , Napačnih: 122 , Točnost: 92.17447081462477
    #   - Ucni vzorci - Pravilnih: 6202 , Napačnih: 36 , Točnost: 99.4228919525489
    # 2) 100 skritih, 0.05 faktor, 100000 iteracij (random, 5 krat vsak vzorec): 
    #   - Testni vzorci - Pravilnih: 1424 , Napačnih: 135 , Točnost: 91.34060295060937
    #   - Ucni vzorci - Pravilnih: 6199 , Napačnih: 39 , Točnost: 99.3747996152613
    # 3) 100 skritih, 0.05 faktor, 100000 iteracij (random, 10 krat vsak vzorec):
    #   - Testni vzorci - Pravilnih: 1441 , Napačnih: 118 , Točnost: 92.43104554201412
    #   - Ucni vzorci - Pravilnih: 6183 , Napačnih: 55 , Točnost: 99.11830714972749
    # 4) 26 skritih, 0.05 faktor, 100000 iteracij (random, 10 krat vsak vzorec):
    #   - Testni vzorci - Pravilnih: 1394 , Napačnih: 165 , Točnost: 89.41629249518923
    #   - Ucni vzorci - Pravilnih: 6048 , Napačnih: 190 , Točnost: 96.95415197178583
    # 5) 617 skritih, 0.05 faktor, 100000 iteracij (random, 5 krat vsak vzorec):
    #   - Testni vzorci - Pravilnih: 1373 , Napačnih: 186 , Točnost: 88.06927517639512
    #   - Ucni vzorci - Pravilnih: 6147 , Napačnih: 91 , Točnost: 98.54119910227637

    ### Program ###

    # Test #
    # Testni vzorci
    def Test_testni(konec):
        pravilnih = 0
        napacnih = 0
        tocnost = 0

        for i in range(len(testniPodatki)):

            data = testniPodatki[i]
            dolzina = len(data)
            znacilke = data[0:(dolzina-1)]
            razred = data[(dolzina-1)]
            rezredPredict = 0

            testData = znacilke.reshape(-1, 1)
            rezultat = perc.FeedForward(testData) # Testiranje mreže

            for i in range(26):
                rezredPredict += rezultat[i][0] * (i + 1)

            if rezredPredict == razred:
                pravilnih += 1
            else:
                napacnih += 1

        tocnost = (pravilnih / len(testniPodatki)) * 100

        if konec == 1: print("Testni vzorci - Pravilnih:", pravilnih, ", Napačnih:", napacnih, ", Točnost:", tocnost)

        return tocnost


    # Učni vzorci
    def Test_ucni():
        pravilnih = 0
        napacnih = 0
        tocnost = 0

        for i in range(len(ucniPodatki)):

            data = ucniPodatki[i]
            dolzina = len(data)
            znacilke = data[0:(dolzina-1)]
            razred = data[(dolzina-1)]
            rezredPredict = 0

            testData = znacilke.reshape(-1, 1)
            rezultat = perc.FeedForward(testData) # Testiranje mreže

            for i in range(26):
                rezredPredict += rezultat[i][0] * (i + 1)

            if rezredPredict == razred:
                pravilnih += 1
            else:
                napacnih += 1

        tocnost = (pravilnih / len(ucniPodatki)) * 100

        print("Ucni vzorci - Pravilnih:", pravilnih, ", Napačnih:", napacnih, ", Točnost:", tocnost)



    # Učenje #
    tocnostOld = 0
    slabse = 0
    konec = 0

    for j in range(epohe):

        np.random.shuffle(ucniPodatki) # random razmeči podatke
        print("Moja koda - Epocha:", j)

        for i in range(len(ucniPodatki)):
            #izberi = rnd.sample(range(0,6238), 1)
            data = ucniPodatki[i]
            dolzina = len(data)
            znacilke = data[0:(dolzina-1)]
            razred = data[(dolzina-1)]
            razredReal = np.zeros(26)
            razredReal[int(razred-1)] = 1
            
            perc.Ucenje(znacilke.reshape(-1, 1),razredReal.reshape(-1, 1))

        # test prenaučenosti
        naucenost = Test_testni(konec)

        if tocnostOld < naucenost:
            tocnostOld = naucenost
            slabse = 0
        else:
            slabse += 1
            if slabse == 3:
                break
                

    # Izpis točnosti
    konec = 1
    none = Test_testni(konec)
    Test_ucni()
        
    
    """

    ############################
    ### PERCEPTRON - PYTORCH ###
    ############################

    # Preverjanje ali je na voljo GPU (coda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Zbirka podatkov
    ucniPodatki = np.genfromtxt(".\Tri-plastni-perceptron\Koda\isolet1+2+3+4.data", delimiter=', ', dtype=float)
    testniPodatki = np.genfromtxt(".\Tri-plastni-perceptron\Koda\isolet5.data", delimiter=', ', dtype=float)

    # Nastavljanje podatkov o mneži (3 plastni perceptron)
    steviloVhodov = len(ucniPodatki[0]) - 1
    steviloSkritih = 200
    steviloIzhodov = 26
    ucniFaktor = 0.001
    ucenjeIteracije = 10000

    model = perceptronPytorch.PerceptronPytorch(steviloVhodov, steviloSkritih, steviloIzhodov)
    loss_fn = nn.MSELoss() 
    model.to(device)
    
    ## Podatki za učenje - različni algoritmi ##
    # 1) Stochastic Gradient Descent (SGD) #
    #optimizer = optim.SGD(model.parameters(), ucniFaktor) # (100000,5) Testni vzorci - Pravilnih: 1405 , Napačnih: 154 , Točnost: 90.12187299550995
    
    # 2) Adam (Adaptive Moment Estimation) #
    # beta1: nadzira odmik prvega momenta (podobno kot zagon), navadno se začne z okoli 0.9. Pomeni, da algoritem ohranja 90% prejšnjega odmika
    # beta2: nadzira odmik drugega momenta (uteži kvadratov gradientov), navadno se začne z okoli 0.999. Pomeni, da algoritem ohranja 99.9% prejšnjega odmika
    # ne rabiš dosti ponovitev, faktor učenja = 0.001
    #optimizer = optim.Adam(model.parameters(), ucniFaktor, betas=(0.9, 0.999)) # Testni vzorci - Pravilnih: 1465 , Napačnih: 94 , Točnost: 93.97049390635023
    
    # 3) 
    optimizer = optim.Adam(model.parameters(), ucniFaktor, betas=(0.9, 0.999))

    # Učenje
    for i in range(5): # število epoch
        for j in range(ucenjeIteracije):

            if (j % 100) == 1:
                print("Pytorch - Epocha:",i,"Iteracija:",j)

            izberi = rnd.sample(range(0,6238), 1)
            data = ucniPodatki[izberi]
            oblika = data.shape
            dolzina = oblika[1]
            znacilke = data[0][0:(dolzina-1)]
            razred = data[0][(dolzina-1)]
            razredReal = np.zeros(26)
            razredReal[int(razred-1)] = 1

            # Prirejanje podatkov za PyTorch
            znacilke = torch.from_numpy(znacilke)
            razredReal = torch.from_numpy(razredReal)

            # Če je GPU naj dela na njem     
            znacilke = znacilke.to(device)
            razredReal = razredReal.to(device)

            
            # Feedforward
            optimizer.zero_grad() # briše gradiente
            out = model(znacilke)
            loss = loss_fn(out, (razredReal).float())

            # Prilagajanje uteži nazaj
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Testiranje 
    model.eval() # preklopi med učenjem in testiranjem

    pravilnih = 0
    napacnih = 0
    tocnost = 0

    with torch.no_grad(): # sklopi računanje gradientov
        for i in range(len(testniPodatki)):

            data = testniPodatki[i]
            dolzina = len(data)
            znacilke = data[0:(dolzina-1)]
            razred = data[(dolzina-1)]
            rezredPredict = 0
            testData = znacilke.reshape(-1, 1)

            znacilke = torch.from_numpy(znacilke)
            znacilke = znacilke.to(device)

            outputs = model(znacilke) # Testiranje mreže
            outputsEnd = outputs.cpu()

            for i in range(26):
                rezredPredict += np.round(outputsEnd[i]) * (i + 1)

            if rezredPredict == razred:
                pravilnih += 1
            else:
                napacnih += 1
            
    tocnost = (pravilnih / len(testniPodatki)) * 100

    print("Testni vzorci - Pravilnih:", pravilnih, ", Napačnih:", napacnih, ", Točnost:", tocnost)
    """
