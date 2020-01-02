import numpy as np


im1 = '004945'
im2 = '004964'
im3 = '005002'

for n in range(3):
    string = 'im' + str(n+1)+'object.txt'
    file1 = open(string, 'r')
    car=[]
    light=[]
    person=[]
    bicycle=[]
    index = 0
    for line in file1:
        name = line.split(':')[0]
        if (name=='car'):
            car.append(index)
        elif name =='traffic light':
            light.append(index)
        elif name =='bicycle':
            bicycle.append(index)
        else:
            person.append(index)
        index+=1
    file1.close()

    string = 'im' + str(n+1)+'com.txt'
    file2 = open(string, 'r')
    index = 0
    cardist = []
    lightdist = []
    persondist = []
    bicycledist = []
    for line in file2:
        dist = [float(s) for s in line.split()]
        if (index in car):
            if not cardist or cardist[0] > np.linalg.norm(dist):
                cardist = (np.linalg.norm(dist), dist[0])
        elif (index in light):
            if not lightdist or lightdist[0] > np.linalg.norm(dist):
                lightdist = (np.linalg.norm(dist), dist[0])
        elif (index in person):
            if not persondist or persondist[0] > np.linalg.norm(dist):
                persondist = (np.linalg.norm(dist), dist[0])
        else:
            if not bicycledist or bicycledist[0] > np.linalg.norm(dist):
                bicycledist = (np.linalg.norm(dist), dist[0])
        index+=1

    if (n==0):
        print ('\nin scene 004945.jpg')
    elif(n==1):
        print ('\nin scene 004964.jpg')
    else:
        print ('\nin scene 005002.jpg')

    print ('number of car is {}'.format(len(car)))
    if (cardist):
        if cardist[1] > 0:
            print ('closest is {} meters away to your right'.format(cardist[0]))
        else:
            print ('closest is {} meters away to your left'.format(cardist[0]))

    print ('number of traffic light is {}'.format(len(light)))
    if (lightdist):
        if lightdist[1] > 0:
            print ('closest is {} meters away to your right'.format(lightdist[0]))
        else:
            print ('closest is {} meters away to your left'.format(lightdist[0]))

    print ('number of person is {}'.format(len(person)))
    if (persondist):
        if persondist[1] > 0:
            print ('closest is {} meters away to your right'.format(persondist[0]))
        else:
            print ('closest is {} meters away to your left'.format(persondist[0]))

    print ('number of bicycle is {}'.format(len(bicycle)))
    if (bicycledist):
        if bicycledist[1] > 0:
            print ('closest is {} meters away to your right\n'.format(bicycledist[0]))
        else:
            print ('closest is {} meters away to your left\n'.format(bicycledist[0]))