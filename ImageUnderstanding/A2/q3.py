import numpy as np
import matplotlib.pyplot as plt



P = .99
# p = 0.7
k = 5



def f(p):
    return   (1/2)*(p)**2




t1 = np.arange(0.1, 0.6, 0.1)
t2 = np.arange(0.1, 0.6, 0.02)
print (f(t1))
plt.figure(1)
# plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')


# A = f(t1).astype('int')


# for x,y in zip(t1,A):
#     plt.annotate((x,y), xy=(x,y+y*0.5))

plt.xlabel('k', fontsize=18)
plt.ylabel('S', fontsize=16)

# # plt.ylim(top =1)
# # plt.ylim(bottom =-1)
# plt.xticks(np.arange(0, 7,1))

plt.show()

# P = 0.99
# p = 0.4
# k = 4
# # k = 4 #for homography

# S = np.log(1-P) / np.log(1 - p**k)
# print "number of iteration need is: " , S

# a = [1,2,3]
# print(a[-1])