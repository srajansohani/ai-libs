class Matrix:
    def __init__(self,rows,columns,values=None):
        self.rows = rows
        self.columns = columns
        self.data = []
        self.initialize_matrix(values)

    def initialize_matrix(self,values):
        if(values is None or (len(values) != self.rows or len(values[0]) != self.columns)):
            for i in range(self.rows):
                row = []
                for j in range(self.columns):
                    row.append(0)
                self.data.append(row)
        else:
            for i in range(self.rows):
                row = []
                for j in range(self.columns):
                    row.append(values[i][j])
                self.data.append(row)


    @staticmethod
    def multiply(matrix1,matrix2):
        if(matrix1.columns != matrix2.rows):
            raise Exception("Matrix multiplication cannot occur as no of column in first matrix is not equal to no of rowsin second matrix" )

        result = Matrix(matrix1.rows,matrix2.columns)
        for i in range(matrix1.rows):
            for j in range(matrix2.columns):
                for k in range(matrix1.columns):
                    result.data[i][j] += matrix1.data[i][k] * matrix2.data[k][j]
        
        return result
    
    @staticmethod
    def add(matrix1,matrix2):
        if(matrix1.rows != matrix2.rows or matrix1.columns != matrix2.columns):
            raise Exception("Matrix addition cannot occur as no of rows and columns in both matrices are not equal" )

        result = Matrix(matrix1.rows,matrix1.columns)
        for i in range(matrix1.rows):
            for j in range(matrix1.columns):
                result.data[i][j] = matrix1.data[i][j] + matrix2.data[i][j]
        
        return result
    
    @staticmethod
    def subtract(matrix1,matrix2):
        if(matrix1.rows != matrix2.rows or matrix1.columns != matrix2.columns):
            raise Exception("Matrix subtraction cannot occur as no of rows and columns in both matrices are not equal" )

        result = Matrix(matrix1.rows,matrix1.columns)
        for i in range(matrix1.rows):
            for j in range(matrix1.columns):
                result.data[i][j] = matrix1.data[i][j] - matrix2.data[i][j]
        
        return result

    @staticmethod
    def copy(matrix):
        result = Matrix(matrix.rows,matrix.columns)
        for i in range(matrix.rows):
            for j in range(matrix.columns):
                result.data[i][j] = matrix.data[i][j]
        
        return result
    
        
    def transpose(self):
        result = Matrix(self.columns,self.rows)
        for i in range(self.rows):
            for j in range(self.columns):
                result.data[j][i] = self.data[i][j]
        
        return result
    
    def shape(self):
        return [self.rows,self.columns]
    
    def multiply_scalar(self,scalar):
        for i in range(self.rows):
            for j in range(self.columns):
                self.data[i][j] = self.data[i][j] * scalar
        return self
        
            
    @staticmethod
    def makeRow(array):
        row = []
        for i in range(len(array)):
            row.append([array[i]])

        return Matrix(1,len(array),row)