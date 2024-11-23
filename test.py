intervals = [[1,3],[2,6],[8,10],[15,18]]


sorted_arr = sorted(intervals, key=lambda x: x[0])
for i in range(1,len(sorted_arr)-1):
    if sorted_arr[i-1][1] >=sorted_arr[i][0]:
        sorted_arr[i-1][1] = sorted_arr[i][1]
        sorted_arr.pop(i)







print(sorted_arr)