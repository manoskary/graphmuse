
class MatrixCOO(object):
	__slots__=('non_zero_entries','shape')

	def __init__(self, shape):
		self.shape = shape

		self.non_zero_entries = list()

	def __repr__(self):
		cursor = 0
		

		matrix_builder = []

		m_r, m_c, m_v = self.non_zero_entries[cursor]

		for r in range(self.shape[0]):
			row_builder = []
			for c in range(self.shape[1]):
				if m_r == r and m_c == c:
					row_builder.append(str(m_v))

					cursor+=1

					if cursor < len(self.non_zero_entries):
						m_r, m_c, m_v = self.non_zero_entries[cursor]
				else:
					row_builder.append('0')

			matrix_builder.append('\t'.join(row_builder))

		return '\n'.join(matrix_builder)

def dense_to_COO(dense_matrix, epsilon=0):
	m = MatrixCOO(dense_matrix.shape)	

	for r in range(dense_matrix.shape[0]):
		for c in range(dense_matrix.shape[1]):
			if abs(dense_matrix[r][c])>epsilon:
				m.non_zero_entries.append((r,c,dense_matrix[r][c]))

	return m


def multiplyCOO(A, B, epsilon = 0):
	assert A.shape[1]==B.shape[0]

	C = MatrixCOO((A.shape[0], B.shape[1]))

	A_cursor = 0
	B_cursor = 0

	for r in range(C.shape[0]):
		for c in range(C.shape[1]):
			acc = 0

			for k in range(A.shape[1]):
				if A_cursor>=len(A.non_zero_entries) or B_cursor>=len(B.non_zero_entries):


				A_r, A_c, A_v = A.non_zero_entries[A_cursor]
				
				if A_r == r and A_c == k:
					a=A_v
					A_cursor+=1
				else:
					a=0

				B_r, B_c, B_v = B.non_zero_entries[B_cursor]
				if B_r == k and B_c == c:
					b=B_v
					B_cursor+=1
				else:
					b=0

				acc += a*b

			if abs(acc)>epsilon:
				C.non_zero_entries.append((r,c,acc))

	return C


class MatrixCRS(object):
	__slots__=('non_zero_entries_per_row','non_zero_entries','shape')

	def __init__(self, shape):
		self.shape = shape

		self.non_zero_entries_per_row = [0]*shape[0]
		self.non_zero_entries = list()

		

	

	def __repr__(self):
		matrix_builder = []
		cursor=0

		for r in range(self.shape[0]):
			row_builder = []
			
			for c in range(self.shape[1]):
				if cursor>=len(self.non_zero_entries):
					row_builder.append('0')
					continue

				col, val = self.non_zero_entries[cursor]
				if col == c:
					row_builder.append(str(val))

					cursor+=1
				else:
					row_builder.append('0')

			matrix_builder.append('\t'.join(row_builder))

		return '\n'.join(matrix_builder)


def dense_to_CRS(dense_matrix, epsilon=0):
	m = MatrixCRS(dense_matrix.shape)

	for r in range(dense_matrix.shape[0]):
		for c in range(dense_matrix.shape[1]):
			if abs(dense_matrix[r][c])>epsilon:
				m.non_zero_entries_per_row[r]+=1
				
				m.non_zero_entries.append((c,dense_matrix[r][c]))


def multiplyCRS(A, B):
	assert A.shape[1]==B.shape[0]

	C = MatrixCRS((A.shape[0], B.shape[1]))

	for r in range(C.shape[0]):
		for c in range(C.shape[1]):
			


def rand_binary_matrix(shape):
	from numpy.random import random
	rbm = random(shape)

	rbm[rbm<=0.5]=0
	rbm[rbm>0.5]=1

	return rbm


A = rand_binary_matrix((5,7))
B = rand_binary_matrix((7,9))



print(A@B)

print()

print(multiplyCOO(dense_to_COO(A), dense_to_COO(B)))