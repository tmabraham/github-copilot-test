# 这个脚本实现了一个简单的排序算法
# 实现的功能是对一个列表中的数据进行升序排序
# 实现的方法是：遍历列表中的数据，将其与后面的数据比较，若后面的数据小于前面的数据，则将后面的数据放到前面的数据之前，并将后面的数据置为None，即排序完成。
# 在实现这个算法时，需要注意的是：需要将每一个数据都转换为字符串，否则会报错，因为字符串是不可变的类型。

def sort(list):
    for i in range(len(list)):
        for j in range(i,len(list)):
            if list[i] > list[j]:
                list[i],list[j] = list[j],list[i]
    return list

list = [3,2,1,4,5,6,7,8,9]
print(sort(list))

# 实现的方法2：使用Python内置的sorted函数

list = [3,2,1,4,5,6,7,8,9]
print(sorted(list))
print(sorted(list,reverse=True))



