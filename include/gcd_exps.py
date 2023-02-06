def gcd(a,b):
	r1, r2 = a,b

	while r2 > 0:
		tmp = r2
		r2 = r1%r2
		r1 = tmp

	return r1

def totient(n):
	count = 2
	for d in range(2,n-1):
		count+=int(gcd(d,n)==1)

	return count

n = 5

while n < 10000000:
	print(n,totient(n)/n)
	n = int(1.25*n)
	n += int(n%2==0)