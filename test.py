def Sum3(nums, number):
    num = 0
    for i in range(0, len(nums)-2):
        for j in range(i+1, len(nums)-1):
            if number-nums[i]-nums[j] in nums[(j+1):]:
                num += 1
    return num
