import numpy
import numpy as np
x=range(16)
x_np=np.reshape(x,(4,4))
print(x_np)
print("type",type(x_np))
print("shape=", x_np.shape)
print(f"number of rows {x_np.shape [0]},number of columns {x_np.shape [1]}")
#indexing
print(x_np[1],x_np[1])
print(x_np[:,2])
print("row 2 col 5 = ",x_np[1,3])
print("specific columns =",x_np[3,[0,3,2]])'''
#gentraing matrices
'''mat1=np.zeros((5,5))
print(mat1)
mat2=np.ones((5,5))
print(mat2)
mat3=np.full((8,8),8)
print(mat3)'''

'''elements=[[2,3,4,5],[7,8,4,7],[9,7,5,7]]
m1=np.array(elements)
print(m1)
m2=np.array([[2,5,6,5],[5,8,6,3],[8,8,8,8]])
print(m2)'''

#performing calculations but this calculations are wrong
'''print("addition",m1+m2)
print(numpy.add(m1,m2))
print(np.subtract(m1,m2))
print(np.multiply(m1,m2))
print(np.divide(m1,m2))
print("calculations are somewhat done")

# Matrix Multiplication
# (a,b) * (m,n)  = (a,n)
# b should be equal to m only then Matrix Multiplication is possible
# (m,n) * (a,b) ; n should be equal to a  => (m,b)
'''m2=m2.transpose()#transpose is used for exchanging of rows and cols here it become row to col and col to row
print(m2)
print(np.matmul(m1,m2))#this is called orginal right matrix muliplication
print(m1@m2)'''


