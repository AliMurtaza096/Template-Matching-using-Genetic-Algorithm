import random

import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np
import math


bigimg = img.imread("groupGray.jpg")
template = img.imread("boothiGray.jpg")


threshold = 0.85


    

# print(c)

b_shape = bigimg.shape

rows = b_shape[0]
r = rows -1
cols = b_shape[1]
c= cols -1


size = bigimg.size


sizeofpop = 100
gen_max = []
gen_mean = []
gens = []
max_coords = []
max_cor = 0

def initialize_pop(rows,columns,sizeOfpop):
    population = []
    for i in range(sizeOfpop):
        row = random.randint(0,r)

        col = random.randint(0,c)
        point = (row,col)
        population.append(point)
    
    #
    return population

population = initialize_pop(rows,cols,sizeofpop)



def fitness_eval(bigimg,template,population):
    correlation_list = []
    for i in range(len(population)):    
        point = population[i]
        if point[0]+35 < (bigimg.shape)[0] and point[1]+29 < (bigimg.shape)[1]:
            prow,pcol = point[0],point[1]
            sliced_img = bigimg[point[0]:point[0]+35,point[1]:point[1]+29]
            fitness = correlation(template,sliced_img)
            corr = (fitness,prow,pcol)
            correlation_list.append(corr)
        else:
            prow,pcol = point[0],point[1]
            corr = (0,prow,pcol)
            correlation_list.append(corr)

    return correlation_list


    

def correlation(template,sliced_img):
   
    temp_mean = np.mean(template)

    slice_mean = np.mean(sliced_img)

    num = 0
    d1 = 0
    d2 = 0
    slice_col = 0
    for i in range(len(sliced_img)):
        temp =template[i]
        sliced = sliced_img[i]

     
        for j in range(len(temp)):
            if j < len(sliced):
                # if sliced[j] != []:
                slice_value = sliced[j]
                # print(temp[i])
                num +=(temp[j] - temp_mean)*(slice_value - slice_mean)
                d1 += (temp[j] - temp_mean)** 2
                d2 += (slice_value - slice_mean)**2
                a = d1 * d2 
    
    denum = math.sqrt(a)
    corr = num / denum
    return corr


def rankedPop(pop_correlation):
    pop_correlation.sort(key = lambda x: x[0],reverse = True)
    

    corr_array = []
    points_array = []
    # print(len(pop_correlation))
    for i in range(len(pop_correlation)):
        point = pop_correlation[i]
        corr_array.append(point[0])
        coordinate = (point[1],point[2])
        points_array.append(coordinate)
    b= corr_array[0]
    max_cor = b

    
    gen_max.append(b)
    mean = sum(corr_array) /len(corr_array)

    max_point = points_array[0]
    max_coords.append(max_point)


    points_array[(len(points_array)-1)] = points_array[0]
            


    gen_mean.append(mean)
    return points_array, max_cor




def crossover(sorted_pop):
    # print(sorted_pop)
    bimg_row_size = len(bin(b_shape[0]-1).replace("0b",""))

    bimg_col_size = len(bin(b_shape[1]-1).replace("0b",""))
  
    
    new_pop = []
    n1_ltemp = 0
    n1_rtemp = 0
    n2_ltemp = 0
    n2_rtemp = 0
    for i in range(len(sorted_pop)):
        if i % 2 == 0:
            i_point = sorted_pop[i]
            n1_row = bin(i_point[0]).replace("0b","") 
            n1_col = bin(i_point[1]).replace("0b","")
            
            n1_row = n1_row.zfill(bimg_row_size)

            
            n1_col = n1_col.zfill(bimg_col_size) 

            n1  = n1_row + n1_col
            


            i_point2 = sorted_pop[i+1]
            n2_row = bin(i_point2[0]).replace("0b","") 
            n2_col = bin(i_point2[1]).replace("0b","") 

            n2_row = n2_row.zfill(bimg_row_size)

            n2_col = n2_col.zfill(bimg_col_size) 

            
            n2 = n2_row + n2_col



            slicer = random.randint(0,len(n1))
            # slicer = 5
            
            n1_ltemp = n1[:slicer]
            n1_rtemp = n1[slicer:]

            n2_ltemp = n2[:slicer]
            n2_rtemp = n2[slicer:]


            c1_bin = n1_ltemp +n2_rtemp
            c2_bin = n2_ltemp + n1_rtemp
            # print(n1, c1_bin,"\n", n2,c2_bin )
        



            c1_row = c1_bin[:(len(c1_bin)) // 2]
            c1_col = c1_bin[(len(c1_bin) // 2):]

            c2_row = c2_bin[:(len(c2_bin)) // 2]
            c2_col = c2_bin[(len(c2_bin) // 2):]

            c1 = (int(c1_row,2), int(c1_col, 2))
            c2 = (int(c2_row,2), int(c2_col, 2))
            new_pop.extend([c1,c2])
    
    return new_pop

#
def mutation(crossed_pop,max_cor):
    # print(len(crossed_pop))
    mutated_pop = []
    rand_indexes = []
    for i in range(0,1):
        
        rand = random.randint(2,98)
        rand_indexes.append(rand)
    # print(rand_indexes)

    for i in (rand_indexes):
        

        cross_point = crossed_pop[i]
        row = cross_point[0]
        col = cross_point[1]
        # print(row,col)
        row_bin = bin(row).replace("0b","")
        col_bin = bin(col).replace("0b","")
        
       
        mutated_row_bin = ''
        mutated_col_bin = ''
        mutated_col_bin = col_bin

        

        for i in range(len(row_bin)):
            if i != 0:
                # print("bbbbbbbbb")
                mutated_row_bin += row_bin[i] 
            else:
                if row_bin[i] == '0':
                    # print('aaaaaaaaaaaaaaaaaa')
                    mutated_row_bin += str(1)
                else:
                    mutated_row_bin += str(0)
        

             

        mutated_point = mutated_row_bin + mutated_col_bin
        mutated_row = mutated_point[:len(mutated_point) // 2]
        mutated_col = mutated_point[(len(mutated_point)) //2 :]

        mutated_p = (int(mutated_row,2),int(mutated_col,2))
        crossed_pop[i]=mutated_p

        # print("sadadsadasd")
    return crossed_pop
# population = mutation(crossed_pop)


# print(gen_mean)

max_counter = 0
g = 0
gen = 0
pre_max = 0
while gen < 100 and threshold > max_cor and max_counter != 600  :
    print(max_cor)
    g +=1
    pop_correlation = fitness_eval(bigimg,template,population)
    # print(pop_correlation)
    sorted_pop,max_cor = rankedPop(pop_correlation)
    if pre_max == max_cor:
        max_counter +=1
    else:
        max_counter =0
    crossed_pop= crossover(sorted_pop)
    # population= crossover(sorted_pop)
    # print(crossed_pop)
    population = mutation(crossed_pop,max_cor)

    gens.append(g)
    # print("ccccccccccc")
    gen +=1
    pre_max = max_cor
    print(gen,max_counter)
# np.corrcoef(template)

maxi =max(gen_max)
mp= max_coords[len(max_coords)-1]





im = Image.open('groupGray.jpg')


plt.imshow(im)

ax = plt.gca()


rect = Rectangle((mp[1],mp[0]), 29, 35, linewidth=1, edgecolor='r', facecolor='none')

ax.add_patch(rect)

plt.show()









# importing the required module
# import matplotlib.pyplot as plt
plt.plot(gens,gen_max , label = "Max")
plt.plot(gens,gen_mean, label = "Mean")
 

# naming the x axis
plt.xlabel('Generations')
# naming the y axis
plt.ylabel('Correlation Value')


# giving a title to my graph
plt.title('Babay ki Boothy!')
 
# show a legend on the plot
plt.legend()
 
# function to show the plot
plt.show()











        
    
    
