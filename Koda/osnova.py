import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from multiprocessing import Pool
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
    steviloSkritih = 200
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

    # Test z epohami (200, 0.05, 37 epoh)
    # Testni vzorci - Pravilnih: 1453 , Napačnih: 106 , Točnost: 93.20076972418217
    # Ucni vzorci - Pravilnih: 6238 , Napačnih: 0 , Točnost: 100.0

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

        return tocnost, napacnih


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
    tocnostGraf = []
    napakeGraf = []

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
        naucenost, napake = Test_testni(konec)
        tocnostGraf.append(naucenost)
        napakeGraf.append(napake)

        if tocnostOld <= naucenost:
            tocnostOld = naucenost
            slabse = 0
        else:
            slabse += 1
            if slabse == 5:
                break
                

    # Izpis točnosti
    konec = 1
    none = Test_testni(konec)
    Test_ucni()

    plt.plot(tocnostGraf)
    plt.title('Točnost glede na epohe')
    plt.xlabel('Epohe')
    plt.ylabel('Točnost [%]')
    plt.show()
    plt.plot(napakeGraf)
    plt.title('Napake glede na epohe')
    plt.xlabel('Epohe')
    plt.ylabel('Št. napak')
    plt.show()

        
    
    """

    ############################
    ### PERCEPTRON - PYTORCH ###
    ############################

    # Preverjanje ali je na voljo GPU (coda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    print("Device:", device)

    # Zbirka podatkov
    ucniPodatki = np.genfromtxt(".\Tri-plastni-perceptron\Koda\isolet1+2+3+4.data", delimiter=', ', dtype=float)
    testniPodatki = np.genfromtxt(".\Tri-plastni-perceptron\Koda\isolet5.data", delimiter=', ', dtype=float)

    # Nastavljanje podatkov o mneži (3 plastni perceptron)
    steviloVhodov = len(ucniPodatki[0]) - 1
    steviloSkritih = 200
    steviloIzhodov = 26
    ucniFaktor = 0.01
    epohe = 100

    model = perceptronPytorch.PerceptronPytorch(steviloVhodov, steviloSkritih, steviloIzhodov)
    loss_fn = nn.MSELoss() 
    model.to(device)


    # Testiranje #
    # Testni
    def Test_testna(konec):
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
                #testData = znacilke.reshape(-1, 1)

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

        if konec == 1: print("Testni vzorci - Pravilnih:", pravilnih, ", Napačnih:", napacnih, ", Točnost:", tocnost)

        return tocnost, napacnih

    # Ucni
    def Test_ucna():
        model.eval() # preklopi med učenjem in testiranjem

        pravilnih = 0
        napacnih = 0
        tocnost = 0

        with torch.no_grad(): # sklopi računanje gradientov
            for i in range(len(ucniPodatki)):

                data = ucniPodatki[i]
                dolzina = len(data)
                znacilke = data[0:(dolzina-1)]
                razred = data[(dolzina-1)]
                rezredPredict = 0
                #testData = znacilke.reshape(-1, 1)

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
                
        tocnost = (pravilnih / len(ucniPodatki)) * 100

        print("Učni vzorci - Pravilnih:", pravilnih, ", Napačnih:", napacnih, ", Točnost:", tocnost)

    
    ## Podatki za učenje - različni algoritmi ##
    #! Za uporabo posamezne funkcije odkomentiraj vrstici optimiter in ponovi !#
        
    # 1) Stochastic Gradient Descent (SGD) #
    # faktro učenja = 0.1, ima dosti ponovitev
    optimizer = optim.SGD(model.parameters(), ucniFaktor)
    ponovi = 5
    # Testni vzorci - Pravilnih: 1428 , Napačnih: 131 , Točnost: 91.59717767799872
    # Učni vzorci - Pravilnih: 5971 , Napačnih: 267 , Točnost: 95.7197819814043
    
    # 2) Adam (Adaptive Moment Estimation) #
    # beta1: nadzira odmik prvega momenta (podobno kot zagon), navadno se začne z okoli 0.9. Pomeni, da algoritem ohranja 90% prejšnjega odmika
    # beta2: nadzira odmik drugega momenta (uteži kvadratov gradientov), navadno se začne z okoli 0.999. Pomeni, da algoritem ohranja 99.9% prejšnjega odmika
    # ne rabiš dosti ponovitev, faktor učenja = 0.001
    #optimizer = optim.Adam(model.parameters(), ucniFaktor, betas=(0.9, 0.999))
    #ponovi = 3
    # Testni vzorci - Pravilnih: 1461 , Napačnih: 98 , Točnost: 93.71391917896086
    # Učni vzorci - Pravilnih: 6216 , Napačnih: 22 , Točnost: 99.64732285989099
    
    # 3) RMSprop (Root Mean Square Propagation)
    # alpha: gladilna konstanta. Ponavadi okrog 1 (0.99 ali 0.95)
    # eps: vrednost dodana v imenovalec da preprečiš deljenje z 0
    # weight_decay: manjšanje uteži in pomaga pred overfittingom
    # momentum: moment da pohitri učenje
    # centered: compute the centered RMSProp, the gradient is normalized by an estimation of its variance.
    # ucni faktor = 0.01
    #optimizer = optim.RMSprop(model.parameters(), ucniFaktor, alpha=0.99, eps=1e-08, weight_decay=0.01, momentum=0, centered=False)
    #ponovi = 1
    # Testni vzorci - Pravilnih: 60 , Napačnih: 1499 , Točnost: 3.8486209108402822
    # Učni vzorci - Pravilnih: 240 , Napačnih: 5998 , Točnost: 3.847386983007374

    # 4) Adagrad (Adaptive Gradient Algorithm)
    # lr_decay: Learning rate decay.
    # weight_decay: Weight decay (L2 penalty) can help prevent overfitting.
    # initial_accumulator_value: Starting value for the accumulators, which hold the sum of squares of gradients.
    # eps: Term added to the denominator to improve numerical stability.
    #optimizer = optim.Adagrad(model.parameters(), ucniFaktor, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
    #ponovi = 5
    # Testni vzorci - Pravilnih: 1462 , Napačnih: 97 , Točnost: 93.7780628608082
    # Učni vzorci - Pravilnih: 6041 , Napačnih: 197 , Točnost: 96.84193651811478

    # Učenje
    tocnostOld = 0
    slabse = 0
    konec = 0 # samo za izpis
    tocnostGraf = []
    napakeGraf = []

    for i in range(epohe): # število epoch

        np.random.shuffle(ucniPodatki) # random razmeči podatke
        print("Pytorch - Epocha:",i) 

        for i in range(len(ucniPodatki)):
            data = ucniPodatki[i]
            dolzina = len(data)
            znacilke = data[0:(dolzina-1)]
            razred = data[(dolzina-1)]
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

        # test prenaučenosti
        naucenost, napake = Test_testna(konec)
        tocnostGraf.append(naucenost)
        napakeGraf.append(napake)

        if tocnostOld <= naucenost:
            tocnostOld = naucenost
            slabse = 0
        else:
            slabse += 1
            if slabse == ponovi:
                break
    
    # Izpis točnosti
    konec = 1
    none = Test_testna(konec)
    Test_ucna()

    plt.plot(tocnostGraf)
    plt.title('Točnost glede na epohe')
    plt.xlabel('Epohe')
    plt.ylabel('Točnost [%]')
    plt.show()
    plt.plot(napakeGraf)
    plt.title('Napake glede na epohe')
    plt.xlabel('Epohe')
    plt.ylabel('Št. napak')
    plt.show()
    """

    

