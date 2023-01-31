#Since I worked on Python Before there might be some extra Solved chalenges 




#Hello World
if __name__ == '__main__':
    print ("Hello, World!")

#Write a function
#I over-ride defualt return to write down a ternary operation(used for simplicity, Google it!)

def is_leap(year):
    
    leap = False
    
    if year % 4 ==0 : 
        leap = True   
         
        if year%100 ==0: 
            
            leap = False
                
            if year%400 ==0:
                leap = True
        
               

    return leap
                    


#Loops


if __name__ == '__main__':
    n = int(raw_input())
    for _ in range(n):  #By convention we use _ if the variable name is not                                important or we are not expecting it to show up                                   somewhere else in our code(not  resuable)
        print(_*_)


#Python: Division


#Defining m&n as our variables, at the time we define variables they became gobaly accessable.
m = n = 0

def integer_division(a,b):
    global n                #we need to call global each time we want to write on                              variable ( it is not necessary to call global to                                  read variable values )
    n = a//b 
    return n

def float_division(a,b):
    global m
    m = a/b
    return m


if __name__ == '__main__':
    a = int(raw_input())
    b = int(raw_input())
    print(integer_division(a,b)) #Passing & calling a,b values to the functions
    print(float_division(a,b))   #Getting m , n as our return vlaues to print


#Arithmetic Operators:
#For some reason this solution rises a syntax error after using sep in the print function; but it completely works fine when you run it locally on your machine, I just wanted to use a solution that could be more scalable in the future for making a calculator and practice some skills, cheers!.


Subs = 0
s = 0
p = 0   


def S(a,b):
    global s
    s = (a+b)
    return s

def sub(a,b):
    global subs
    subs = (a - b)
    return subs

def p(a,b):
    global p
    p = (a*b)
    return p
    



if  __name__ == '__main__':
        
    a = int(input())
    b = int(input())
    # print(S(a,b), sub(a,b),p(a,b),sep='\n') #this won't work in hackerrank compiler
    print(S(a,b))
    print(sub(a,b))
    print(p(a,b))
    

#Python if-else

#!/bin/python

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(raw_input().strip())
    if (n%2==1 or (n%2==0 and 6<=n<=20)) : print("Weird") 
         
    elif (n>20 or n%2==0 and 2<=n<=5): print("Not Weird")
    
#You can put pranthesses between 2 conditions to seperate and make them more readable to others(Hence difference between if and elif syntax)




#Alphabet Rangoli

import sys

stdin = sys.stdin
def print_rangoli(size):
    # size = int(stdin.readline())

    for i in range(2*size-1):
        d = size-1-abs(size-1-i)
        for j in range(4*size-3):
            if j % 2 == 0 and abs((j-(2*size-2))//2) <= d:
                sys.stdout.write(chr(97+size-1-(d-abs((j-(2*size-2))//2))))
            else:
                sys.stdout.write("-")
        sys.stdout.write("\n")



#String Formatting

def print_formatted(number):
# Decimal
# Octal
# Hexadecimal (capitalized)
# Binary
#Python by default supports oct,hex,bin of the numbers we just change input value to string calculate our value and then bring it back to integer.
#xrange is the equivalent of range in python 2.


    maxSpace = len(bin(n))-2

    for i in xrange(1,n+1):
        print (str(i).rjust(maxSpace), str(int(oct(i))).rjust(maxSpace), str(hex(i)[2:])
        .upper()).rjust(maxSpace), str(bin(i)[2:]).rjust(maxSpace)
    




#Merge and tools!

#['AABCAAADA']

def merge_the_tools(string, k):
    s = string
    chars = len(s)/k
    # reFormatedString = ','.join(s)
    i = 0
    
    while i < len(s):
        
        list1 = s[i:i+k]
        list2 = ""
        for x in list1:
            if x not in list2:   
                list2 +=x
                
        print(list2)
        i += k
   
   
    
#The Minion Game
from collections import defaultdict

def minion_game(string):
    n = string.split(" ")
    vowels = ['A', 'E', 'I', 'O', 'U']
    # s = raw_input(string)
    a = 0
    b = 0
    for i, c in enumerate(string):
        if c in vowels:
            b += len(string) - i
        else:
            a += len(string) - i
            
    if a == b:
        print ("Draw")
    elif a > b:
        print ('Stuart {}'.format(a))
    else:
        print ('Kevin {}'.format(b))
        

#Print Function




if __name__ == '__main__':
    n = int(raw_input())
    for i in range(n):
        
        print(i+1 , end="")


#String Split and join


def split_and_join(line):
    line = line.split(' ')
    line = '-'.join(line)
    return line
if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)


#sWAP cASE

def swap_case(s):
    for i in s:
        string =''
        # string == i.upper() if i.islower() else string == i.lower()
        if i.islower():
            i = i.upper()
            string += i
            print(string, end='')
        else:       
            i = i.lower()
            string += i
            print(string, end='')
    return string.rstrip(string[-1]) #for some reason it generates an additional dot which                                           removed by using rStrip method.
    
    

#Finding the percentage

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    
    for key,value in student_marks.items():

        if key == query_name: #Filtering values by comparing given name to keys.
            average = (value[1] + value[2] + value[0])/3 #slicing threw given list
            print('{0:.2f}'.format(average)) #Using string formating to show 2 decimal numbers


#Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    
list1 = []
for x in set(sorted(list(arr))):
  list1.append(x)
print(list1[-2])


  


#Capitalize!


# Complete the solve function below.
def solve(s):
   answer = (' '.join([i.capitalize() for i in s.split(' ')]))
   print(answer)
   return(answer) #Return is used to resolve NoneType Error.
          
#Another way to resolve this issue is using .title only probelm is that .title will change to upper case even the words after a number like 3g => 3G
   
    # fullName = s.title()
    # print(fullName)
    # return fullName

    

#Text Alignment

#Replace all ______ with rjust, ljust or center. 

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))




#String Validators

if __name__ == '__main__':
    s = input()
    print(any([x.isalnum() for x in s]))
    print(any([x.isalpha() for x in s]))
    print(any([x.isdigit() for x in s]))
    print(any([x.islower() for x in s]))
    print(any([x.isupper() for x in s]))
        


#Find a String

def count_substring(string, sub_string):
    count = 0;
    for i in range(0, len(string)):
        count += string.count(sub_string,i,i+len(sub_string));
    return count
    


#Mutations
def mutate_string(string, position, character):
    l = list(string)
    l[position] = character
    string = ''.join(l)
    return string



#What's your name?

#
# Complete the 'print_full_name' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING first
#  2. STRING last
#

def print_full_name(first, last):
  print(f"Hello {first} {last}! You just delved into python.")
  



#No idea!

happy = 0
n,m = list(map(int,input().split()))
arr = list(map(int,input().split()))
A,B = set(map(int,input().split())),set(map(int,input().split()))
#we convert values to lists so we can operate over them with for loops.
#Using set on arrays A,B discards duplicates.
for i in arr:
    if i in A:
        happy+=1
    if i in B:
        happy-=1
print(happy)



#Symmetric Difference



M = int(input())
a = set(map(int, input().split()))
N = int(input())
b = set(map(int, input().split()))
diff = a.difference(b)
for i in b.difference(a):
    diff.add(i)
print("\n".join(map(str,sorted(diff))))

#I did try to do this by making two seperate diffs (diff1,diff2) and add them to a third diff and print it out but for some reason it did not worked so I got a bit of help to do for loop from discussions.



#Introduction to Sets

def average(array):
    arr = set(array)
    average = sum(arr)/(len(arr))
    return("%.3f" %average)
    


#Designer Door Mat

N,M=map(int,input().split())
for i in range (N):
    if i <=(N//2)-1:
        s=int(1+(2*i))                 # '.|.' pattern
        t=int((M-(s*3))/2)             # '-' pattern
        print('-'*t+'.|.'*s+'-'*t)
    elif i==N//2:
        s=int((M-7)/2)
        print('-'*s+'WELCOME'+'-'*s)   # WELCOME LINE
    else:
        t=int(2*(N-i)-1)               # '.|.' pattern
        s=int((M-(3*t))/2)             # '-' pattern
        print('-'*s+'.|.'*t+'-'*s)



#Text Wrap


def wrap(string, max_width):
    return textwrap.fill(string,max_width)
   


#Set.difference() Operations

n = input()
en = set(map(int,input().split()))

b = input()
fr = set(map(int,input().split()))

print(len(en-fr))


#Set.intersection() Operations

n = input()
en = set(map(int,input().split()))
b = input()
fr = set(map(int,input().split()))
print(len(en.intersection(fr)))


#Set.union() Operation

n = input()
en = set(map(int,input().split()))
b = input()
fr = set(map(int,input().split()))
print(len(set(en.union(fr))))


#Set.discard(), .remove() & pop()

#I did use reaveal button to solve this one cause I did not understood I suppose to receive the operation to fix the issue, plus this code only works on python 3.
n = input()
s = set(map(int, input().split()))
for i in range(int(input())):
    c = input().split()
    if c[0] == 'pop':
        s.pop()
    elif c == 'remove':
        s.remove(int(c[1]))
    else:
        s.discard(int(c[1]))
print(sum(s))


#Set .add()

N = int(input())

print(len(set([input() for i in range(N)])))



#List Comprehensions

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())


    print([ [i, j, k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if i+j     +k !=n ])




#Check Subset

for i in range(int(input())): #using T input as a iterator to catch new values for sets.
    AE = input()
    A = set(map(int,input().split()))
    BE = input()
    B = set(map(int,input().split()))

    if len(A.difference(B)) == 0: #if the length of the result set difference between two sets be zero it means that the first one (A) is a subset of the second one (B)
        print(True)
    else:
        print(False)

    

#The Capitan's Room

# A Counter is a dict subclass for counting hashable objects. It is a collection where elements are stored as dictionary keys and their counts are stored as dictionary values. Counts are allowed to be any integer value including zero or negative counts. 

from collections import Counter


A = int(input())

# Return a list of the n most common elements and their counts from the most common to the least. If n is omitted or None, most_common() returns all elements in the counter. Elements with equal counts are ordered in the order first encountered:
print(Counter(input().split()).most_common()[-1][0])


#Set Mutations

a,A=int(input()),set(input().split())
n=int(input())
for i in range(n):
    op=input().split()
    l=set(input().split())
    if op[0]=='intersection_update':
        A.intersection_update(l)
    elif op[0]=='update':
        A.update(l)
    elif op[0]=='symmetric_difference_update':
        A.symmetric_difference_update(l)
    elif op[0]=='difference_update':
        A.difference_update(l)
print(sum(map(int,A)))



#Set .symmetric_difference() Operations

n = input()
en = set(map(int,input().split()))
b = input()
fr = set(map(int,input().split()))

print(len(en.symmetric_difference(fr)))


#Check Strict SuperSet

A = set(map(int, input().split()))
B = [set(map(int, input().split())) for i in range(int(input()))]

result = all([A.issuperset(b) and len(A) > len(b) for b in B])

print(result)


#Tuples

if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    t=tuple(integer_list)
    print(hash(t))


#Re.start() & Re.end()

import re
S,k=input(),input()
m = re.search(k,S)
pattern = re.compile(k) #used to compile a custom pattern as a regex pattern
if not m:
    print("(-1, -1)")
while m:
    print(f"({m.start()}, {m.end()-1})")
    m = pattern.search(S,m.start()+1) #searching with the custom regex pattern


#Re.findall() & Re.finditer()
import re
S = input()
f = re.findall(r'(?<=[qwrtypsdfghjklzxcvbnm])([aeiou]{2,})(?=[qwrtypsdfghjklzxcvbnm])',S,re.I)
if f == []:
    print(-1)
else:
    for i in f:
        print(i)
        
#I did not solve this issue by myself


#Nested Lists

if __name__ == '__main__':
    nested_list = []
    score_ = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        nested_list.append([name,score])
        score_.append(score)
    nested_list.sort()
    score_=sorted(set(score_))
    for i in nested_list:
        if i[1] == score_[1]:
            print(i[0])


#Collections.namedtuple()

from collections import namedtuple

N = int(input())

Student = namedtuple('Student',input().split())
spreadsheet = [Student(*input().split()) for _ in range(N)] 
# "*" is to omit Error for number of arguments.

print(sum(int(i.MARKS) for i in spreadsheet)/N)

# sums = sum(Student.MARKS)/int(ID)
# print(Student.MARKS)
# for i in range(N):
#     students = Student()
# students = Student(ID=12,MARKS=20,NAME='DAVID',CLASS='021')



#Word Order

from collections import OrderedDict

#we made an orderedDict and add int(+1) each time we have a word found in our list.
d = OrderedDict()
for _ in range(int(input())):
    word = input()
    d[word] = d.get(word, 0) + 1
    
print(len(d))
print(*d.values())


#Collections.OrderedDict()

from collections import OrderedDict

N = int(input())
MyDict = OrderedDict()
for i in range(N):
    item_name,net_price= input().rsplit(' ',1)
    MyDict[item_name] = MyDict.get(item_name, 0) + int(net_price)
for k,v in MyDict.items():
    print(k,v)
    

    
#DefaultDict Tutorial

from collections import defaultdict

n,m = map(int,input().split())
a = defaultdict(list)
for i in range(n):
    a[input()].append(i+1)
for i in range(m):
    print(*a.get(input(),[-1]))



#Collections.Counter()
from collections import Counter

_ = input()

listOfShoes = Counter(map(int,input().split()))
res = 0

for _ in range(int(input())):
    size,money = map(int,input().split())
    if listOfShoes.get(size):
        listOfShoes[size]-=1
        res+=money
print(res)


#Input()

x,k = map(int,input().split())
print(eval(input()) == k)


#Zipped!

n, x = map(int, input().split())
z = zip(*[list(map(float, input().split())) for _ in range(x)])
print(*[sum(i)/len(i) for i in z], sep='\n')


#Exceptions
for i in range(int(input())):
    try:
        a,b = map(int,input().split())
        print(int(a/b))
    except ZeroDivisionError as e:
        print("Error Code: integer division or modulo by zero")
    except ValueError as e:
        print("Error Code:", e)


#Calendar Module

from calendar import weekday
m, d, y = list(map(int, input().split()))
NumDay = weekday(y,m,d)
print("MONDAY" if NumDay == 0 else "TUESDAY" if NumDay == 1 else "WEDNESDAY" if NumDay == 2 else "THURSDAY" if NumDay == 3 else "FRIDAY" if NumDay == 4 else "SATURDAY" if NumDay == 5 else "SUNDAY")


#Collections.deque()

from collections import deque
d = deque()
N = int(input())
for _ in range(N):
    
    cmd = list(map(str, input().split()))
    if 'append' in cmd:
        d.append(cmd[-1])
    elif 'appendleft' in cmd:
        d.appendleft(cmd[-1])
    elif 'pop' in cmd:
        d.pop()
    elif 'popleft' in cmd:
        d.popleft()

for ele in d:
    print(str(ele), end=" ")
    
#Have to work on eval in python.
# from collections import deque 
# d = deque()
# for _ in range(int(input())):
#         if ' ' in (cmd:= input()):
#                 eval(f'd.{cmd.split()[0]}({int(cmd.split()[1])})')
#            above line is => eval(d.append Or d.appendleft(int))
#         else: 
#                 eval(f'd.{cmd}()')
#                 eval for pop and popleft which do not need integer.
# print(*d)
    
    
#Shape and Reshape

import numpy as np
print(np.array(input().split(),int).reshape(3,3))


#Arrays


def arrays(arr):
    arr1 = numpy.array(arr,float)
    return numpy.flip(arr1)



#Standaradize Mobile Number Using Decorators

def wrapper(f):
    def fun(l):
        fun = f([f'+91 {i[-10:-5]} {i[-5:]}' for i in l])
    return fun



#Map and Lambda Function

cube = lambda f: f**3

def fibonacci(n):
    value = 0
    step = 1

    for _ in range(n):
        yield value
        value, step = step, value + step


#Python Evaluation

eval(input())



#Array Mathematics

import numpy as np

n,m = map(int,input().split())
a = np.array([list(map(int,input().split())) for i in range(n)], int)
b = np.array([list(map(int,input().split())) for i in range (n)],int)
print(a+b)
print(a-b)
print(a*b)
print(a//b)
print(a%b)
print(a**b)


#Eye and identity

import numpy as np
np.set_printoptions(legacy='1.13')


n,m = map(int,input().split())

print(np.eye(n,m,k=0))


#Zeros and Ones

import numpy as np

inputs = tuple(map(int,input().split())) #assigning unpacked values to tuples.

print(np.zeros(inputs,int))
print(np.ones(inputs,int))




#Concatenate
import numpy as np

N,M,P = map(int,input().split())

N_columns = []
M_columns = []

for i in range(N):
    N_columns.append(list(map(int,input().split())))

for _ in range(M):
    M_columns.append(list(map(int,input().split())))
    
N_columns = np.array(N_columns)
M_columns = np.array(M_columns)
    
print(np.concatenate((N_columns,M_columns),axis=0))



#Transpose and Flatten
import numpy as np

N,M = map(int ,input().split())

arr = []
for i in range(N):
    arr.append(list(map(int,input().split())))

arr = np.array(arr)
print(np.transpose(np.array(arr)))
print(arr.flatten())



#Inner and Outer

import numpy as np

A = np.array(input().split(), int)
B = np.array(input().split(), int)

print(np.inner(A, B))
print(np.outer(A, B))


#Dots and Cross

import numpy as np

n = int(input())

A = np.array([list(map(int,input().split())) for i in range (n)],int)

B = np.array([list(map(int,input().split())) for i in range (n)],int)

print(np.matmul(A,B))


#Mean, Var, and Std

import numpy as np

n,m = map(int,input().split())
M = np.array([list(map(int,input().split())) for i in range (n)],int)

tasks = [np.mean(M,axis=1),np.var(M,axis=0),round(np.std(M,axis=None),11)]

for i in tasks:
    print(i)


#Floor, Ceil and Rint

import numpy as np
np.set_printoptions(legacy='1.13')
A = list(map(float,input().split()))

task=[np.floor(A),np.ceil(A),np.rint(A)] #just to make it a bit cleaner

for i in task:
    print(i)
    
    
# print(np.floor(A))
# print(np.ceil(A))
# print(np.rint(A))


#Number Line Jumps


#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#

def kangaroo(x1, v1, x2, v2):
    n = 0

    while n <= 10000:
        if (x1+v1) == (x2+v2):
            print("YES")
            break
        elif (x1+v1) != (x2+v2):
            x1+= v1
            x2+= v2
            n+= 1
            if n == 10001:
                print("NO")
                break
                
            
            
    
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()






#Birthday Cakde Candles

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
    i = max(candles)
    return candles.count(i)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()


#Min and Max

import numpy as np

n,m = map(int,input().split(" "))

p = np.array([list(map(int,input().split())) for i in range (n)],int)

print(np.max(np.min(p,axis = 1)))


#Linear Algebra

import numpy as np

N = int(input())
A = np.array([input().split() for _ in range(N)], float)

print(round(np.linalg.det(A), 11))


#Polynomial

import numpy as np

P = np.array(input().split(), float)
print(np.polyval(P, float(input())))


#Insertion Sort - Part2


#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort2' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort2(n, arr):
    for start in range(1,n):
        # j=start
        while arr[start-1]>arr[start] and start>0:
            arr[start],arr[start-1]=arr[start-1],arr[start]
            start-=1
        print(*arr)
        
        
if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)




#Insertion Sort - Part1

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort1(n, arr):
    for i in range((n-1),0,-1):
        if arr[i] < arr[i-1]:
            tmp = arr[i]
            arr[i] = arr[i-1]
            print(*arr)
            arr[i-1] = tmp
    print(*arr)

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)



#Recursive Digit Sum

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k
#

def superDigit(n, k):
    n1 = sum([int(i) for i in n])
    n1 = str(n1*k)
    while(len(n1) != 1) :
        a = sum([int(i) for i in n1])
        n1 = str(a)
    return n1

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()


#Save the Prisoner

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'saveThePrisoner' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER m
#  3. INTEGER s
#

def saveThePrisoner(n, m, s):
    return (m - 1 + s) % n or n
#I found a way to do it my self but since in the discussion section someone mentioned that this way is not optimize, since we may have millions of candies i worked to understand their way of making an algorithem which is way more optimized for bigger values.
                    
                
        

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input().strip())

    for t_itr in range(t):
        first_multiple_input = input().rstrip().split()

        n = int(first_multiple_input[0])

        m = int(first_multiple_input[1])

        s = int(first_multiple_input[2])

        result = saveThePrisoner(n, m, s)

        fptr.write(str(result) + '\n')

    fptr.close()



#Viral Advertising

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#

def viralAdvertising(n):
    likes = 2
    cumulative = 2
    if n > 1:
        for i in range(2,n+1):
            shares = likes * 3
            likes = shares // 2
            cumulative += likes
    return cumulative
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()


