# nums = [1,2,3,4]
#
# for i in range(len(nums)):
#     for j in range(i + 1, len(nums)):
#         if nums[i] != nums[j]:
#             print(i, j)
# #             print(False)
#
# def karatsuba(x,y):
#     if len(str(x))==1 or len(str(y))==1:
#         return x*y
#     n = max(len(str(x)),len(str(y)))
#     k = n//2
#
#     # x = x1*10**k+x2
#     x1=x//(10**k)
#     x0=x%(10**k)
#     y1 = y // (10 ** k)
#     y0 = y % (10 ** k)
#
#     z2 = karatsuba(x1,y1)
#     z0 = karatsuba(x0,y0)
#     z1 = (x1+x0)*(y1+y0)-z2-z0
#
#     z= z2*(10**n)+z1*(10**k)+z0
#     return z
#
# print(karatsuba(3141592653589793238462643383279502884197169399375105820974944592,2718281828459045235360287471352662497757247093699959574966967627))

s=[[2,3],[1,2],[3,5]]
for i in s:
    print(i)
print(s.sort())